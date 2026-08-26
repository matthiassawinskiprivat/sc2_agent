"""Shared rules for SC2 create_game player slots (Bot vs Computer)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from s2clientprotocol import sc2api_pb2 as sc_pb

DEFAULT_EVERY_K_EPISODES = 100
# SC2 API Status (s2clientprotocol.sc2api_pb2.Status).
SC2_STATUS_LAUNCHED = 1
SC2_STATUS_INIT_GAME = 2
SC2_STATUS_IN_GAME = 3
DEFAULT_IN_GAME_WAIT_SECS = 30.0
DEFAULT_IN_GAME_POLL_SECS = 0.3
# If still ``launched`` this long after join, create/join never took effect.
DEFAULT_LAUNCHED_FAIL_SECS = 5.0
# Lobby settle time before join_game (parallel SC2 startups are flaky).
DEFAULT_POST_CREATE_GAME_DELAY_SECS = 1.0

_STATUS_NAMES = {
    1: "launched",
    2: "init_game",
    3: "in_game",
    4: "in_replay",
    5: "ended",
    6: "quit",
    99: "unknown",
}

# Prefixes aligned with Level0Env / curriculum naming: these maps must not run a
# Computer opponent (extra units, larvae, etc.) or map cleanup cannot empty the game.
MOVEMENT_LEVEL_PREFIXES = (
    "level_0",
    "level_1",
    "level_2",
    "level_3",
    "level_move_",
)
# Build curriculum is solo (no AI opponent); do not seat a Computer player.
BUILD_LEVEL_PREFIXES = (
    "level_build_",
)
# Manager curriculum seats Computer by default (overridable via use_computer).
MANAGER_LEVEL_PREFIXES = (
    "level_manager_",
)


def session_realtime(training_conf: Dict[str, Any]) -> bool:
    """
    Whether ``create_game`` / ``_prepare_start`` use realtime simulation.

    Training and debug-preview envs default to step-based sim (``False``) even when
    ``visualize=True``. Realtime breaks ``debug_create_unit`` / cleanup on multi-unit
    movement levels because ``client.step()`` does not advance the simulation the
    same way. Set ``env.realtime: true`` explicitly for human-facing realtime modes.
    """
    env = training_conf.get("env", {})
    if isinstance(env, dict) and "realtime" in env:
        return bool(env["realtime"])
    if "realtime" in training_conf:
        return bool(training_conf["realtime"])
    return False


def every_k_episodes(training_conf: Dict[str, Any]) -> int:
    """
    Episodes between scheduled hard resets (SC2 relaunch / map rotation).

    Read from flattened ``training_conf['every_k_episodes']`` or nested
    ``training_conf['env']['every_k_episodes']``. Defaults to
    ``DEFAULT_EVERY_K_EPISODES`` when unset.
    """
    env = training_conf.get("env", {})
    if "every_k_episodes" in training_conf:
        k = int(training_conf["every_k_episodes"])
    elif isinstance(env, dict) and "every_k_episodes" in env:
        k = int(env["every_k_episodes"])
    else:
        return DEFAULT_EVERY_K_EPISODES
    if k < 1:
        raise ValueError(f"every_k_episodes must be >= 1, got {k}")
    return k


def use_computer_player(level_name: str, training_conf: Dict[str, Any]) -> bool:
    """
    Whether to add a Computer player alongside the training Bot.

    Default False for movement- and build-style level names; default True for
    manager levels. Override with env.use_computer or flattened
    ``training_conf['use_computer']``.
    """
    env = training_conf.get("env", {})
    if isinstance(env, dict) and "use_computer" in env:
        return bool(env["use_computer"])
    if "use_computer" in training_conf:
        return bool(training_conf["use_computer"])

    if any(level_name.startswith(p) for p in MANAGER_LEVEL_PREFIXES):
        return True

    no_computer = any(
        level_name.startswith(p)
        for p in MOVEMENT_LEVEL_PREFIXES + BUILD_LEVEL_PREFIXES
    )
    return not no_computer


def resolve_computer_difficulty(training_conf: Dict[str, Any]) -> str:
    """YAML ``env.difficulty`` / ``opponent.difficulty`` — never hardcode Easy in callers."""
    from src.runtime.opponent_config import resolve_opponent_config

    opp = resolve_opponent_config(training_conf)
    if opp.get("difficulty"):
        return str(opp["difficulty"])
    env = training_conf.get("env", {})
    if isinstance(env, dict) and env.get("difficulty"):
        return str(env["difficulty"])
    if training_conf.get("difficulty"):
        return str(training_conf["difficulty"])
    return "Easy"


def _status_label(status: Optional[int]) -> str:
    if status is None:
        return "None"
    return f"{status} ({_STATUS_NAMES.get(int(status), 'other')})"


def check_create_game_result(result) -> None:
    """Raise if ``RequestCreateGame`` returned a nested engine error."""
    create = getattr(result, "create_game", None)
    if create is None:
        return
    has_field = getattr(create, "HasField", None)
    if callable(has_field) and has_field("error"):
        err = create.error
        details = ""
        if has_field("error_details"):
            details = f": {create.error_details}"
        raise RuntimeError(f"create_game failed: error={err}{details}")


def check_join_game_result(result) -> None:
    """Raise if ``RequestJoinGame`` returned a nested engine error."""
    join = getattr(result, "join_game", None)
    if join is None:
        return
    has_field = getattr(join, "HasField", None)
    if callable(has_field) and has_field("error"):
        err = join.error
        details = ""
        if has_field("error_details"):
            details = f": {join.error_details}"
        raise RuntimeError(f"join_game failed: error={err}{details}")


def _read_map_bytes(game_map, path: Path) -> Optional[bytes]:
    data = getattr(game_map, "data", None)
    if data:
        return bytes(data)
    if path.is_file():
        return path.read_bytes()
    return None


def _local_map_proto(game_map) -> sc_pb.LocalMap:
    """
    Build LocalMap for create_game.

    SC2 only opens maps under its ``Maps/`` tree by path. Repo maps live under
    ``train/maps/...`` and become absolute ``relative_path`` values — create_game
    may still reach ``init_game``, but ``join_game`` then fails with
    ``LaunchError: Unable to open map``. For those maps, send a filename label
    plus embedded ``map_data`` (never an absolute filesystem path).
    """
    from sc2.paths import Paths

    path = Path(getattr(game_map, "path", getattr(game_map, "relative_path", "")))
    name = str(getattr(game_map, "name", path.stem) or path.stem)
    maps_root = Path(Paths.MAPS).resolve()

    try:
        rel = path.resolve().relative_to(maps_root)
        return sc_pb.LocalMap(map_path=str(rel).replace("\\", "/"))
    except ValueError:
        pass
    except OSError:
        pass

    data = _read_map_bytes(game_map, path)
    if not data:
        raise RuntimeError(
            f"Map {name!r} is outside SC2 Maps/ ({maps_root}) and map bytes "
            f"could not be read from {path}"
        )
    # Filename only — absolute paths break join_game (error=6 Unable to open map).
    return sc_pb.LocalMap(map_path=f"{name}.SC2Map", map_data=data)


async def create_sc2_game(controller, game_map, players: List[Any], realtime: bool):
    """
    RequestCreateGame with nested-error + status checks.

    Unlike stock ``Controller.create_game``, embeds map bytes for absolute paths.
    """
    from sc2.player import Computer

    req = sc_pb.RequestCreateGame(local_map=_local_map_proto(game_map), realtime=realtime)
    for player in players:
        p = req.player_setup.add()
        p.type = player.type.value
        if isinstance(player, Computer):
            p.race = player.race.value
            p.difficulty = player.difficulty.value
            p.ai_build = player.ai_build.value

    result = await controller._execute(create_game=req)
    check_create_game_result(result)
    status = getattr(result, "status", None)
    if status not in {SC2_STATUS_INIT_GAME, SC2_STATUS_IN_GAME}:
        raise RuntimeError(
            f"create_game did not enter init_game (status={_status_label(status)}, "
            f"map={getattr(game_map, 'name', game_map)!r}, path={getattr(game_map, 'path', '?')})"
        )
    return result


async def join_sc2_game(client, race, *, name: Optional[str] = None) -> int:
    """Join as participant; raise on nested join errors. Returns player_id."""
    from sc2.data import Race

    ifopts = sc_pb.InterfaceOptions(
        raw=True,
        score=True,
        show_cloaked=True,
        show_burrowed_shadows=True,
        raw_affects_selection=bool(getattr(client, "raw_affects_selection", False)),
        raw_crop_to_playable_area=False,
        show_placeholders=True,
    )
    assert isinstance(race, Race)
    req = sc_pb.RequestJoinGame(race=race.value, options=ifopts)
    if name is not None:
        req.player_name = str(name)
    result = await client._execute(join_game=req)
    check_join_game_result(result)
    player_id = int(result.join_game.player_id)
    client._game_result = None
    client._player_id = player_id
    status = getattr(result, "status", None)
    if status == SC2_STATUS_LAUNCHED:
        raise RuntimeError(
            f"join_game returned status=launched (player_id={player_id}) — "
            "lobby was not active (create_game likely failed or was cancelled)."
        )
    return player_id


async def host_local_sc2_game(
    *,
    controller,
    client_cls: Type,
    game_map,
    players: List[Any],
    realtime: bool,
    join_race,
    log: Optional[Callable[[str], None]] = None,
    post_create_delay_sec: float = DEFAULT_POST_CREATE_GAME_DELAY_SECS,
    player_name: str = "Bot",
) -> Any:
    """
    Burnysc2-aligned host flow: create_game → fresh Client → join → wait in_game.

    Creating the Client *before* create_game (two Protocol objects on one websocket)
    can leave join stuck in ``launched`` under load. Match ``sc2.main._setup_host_game``.
    """
    _log = log or (lambda _msg: None)

    await controller.ping()
    _log(
        f"Creating game on map={getattr(game_map, 'name', game_map)!r} "
        f"players={', '.join(str(p) for p in players)} realtime={realtime}"
    )
    create_result = await create_sc2_game(controller, game_map, players, realtime)
    _log(f"create_game ok, status={_status_label(getattr(create_result, 'status', None))}")

    if post_create_delay_sec > 0:
        await asyncio.sleep(float(post_create_delay_sec))

    ws = getattr(controller, "_ws", None)
    if ws is None:
        raise RuntimeError("controller has no websocket after create_game")
    client = client_cls(ws)
    player_id = await join_sc2_game(client, join_race, name=player_name)
    raw_status = getattr(client, "_status", None)
    status_val = getattr(raw_status, "value", raw_status)
    _log(f"join_game ok, player_id={player_id}, status={_status_label(status_val)}")

    await wait_for_sc2_in_game(client, log=_log)
    return client


async def wait_for_sc2_in_game(
    client,
    *,
    log: Optional[Callable[[str], None]] = None,
    timeout_sec: float = DEFAULT_IN_GAME_WAIT_SECS,
    poll_sec: float = DEFAULT_IN_GAME_POLL_SECS,
    launched_fail_sec: float = DEFAULT_LAUNCHED_FAIL_SECS,
) -> None:
    """
    Block until SC2 reaches ``in_game`` after ``join_game``.

    SC2 5.0.x often stays in ``init_game`` briefly (worse under multi-instance /
    GPU load). Calling ``get_game_info`` too early raises
    ``ProtocolError: ['A game has not been started yet']``.

    Stuck in ``launched`` means create/join never activated a game — fail fast
    instead of waiting the full timeout.
    """
    _log = log or (lambda _msg: None)
    _log("Waiting for SC2 to transition from init_game → in_game...")
    waited = 0.0
    launched_waited = 0.0
    while waited < timeout_sec:
        ping_resp = await asyncio.wait_for(
            client._execute(ping=sc_pb.RequestPing()), timeout=5.0
        )
        status = getattr(ping_resp, "status", None)
        if status == SC2_STATUS_IN_GAME:
            _log(f"SC2 status = in_game after {waited:.1f}s.")
            return
        if status == SC2_STATUS_LAUNCHED:
            launched_waited += poll_sec
            if launched_waited >= launched_fail_sec:
                raise RuntimeError(
                    "SC2 stuck in launched after join_game — create/join did not "
                    f"start a match. status={_status_label(status)}"
                )
        else:
            launched_waited = 0.0
        _log(f"SC2 status = {_status_label(status)} (waiting for in_game)...")
        await asyncio.sleep(poll_sec)
        waited += poll_sec
    raise RuntimeError(
        f"SC2 did not reach in_game status within {timeout_sec:.0f}s after join_game."
    )

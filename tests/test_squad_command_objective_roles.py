"""CTF squad obs: attack/defend roles must expose command_objective for both players."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.transformers.squad_observation_builder import SquadObservationBuilder
from src.utils.command_constants import CommandConstants
from tests.test_observation_transformer import _make_game_info, create_mock_unit


@pytest.fixture
def training_conf():
    return {
        "agent_architecture": "squad",
        "active_layers": [
            "allied_unit_density",
            "enemy_unit_density",
            "command_objective",
            "command_assignment",
        ],
        "active_tactical_layers": [
            "allied_unit_density",
            "command_objective",
            "command_assignment",
        ],
        "active_unit_scalars": ["role_attack", "role_defend", "dist_to_objective_norm"],
    }


class _UnitsList(list):
    """Minimal stand-in for python-sc2 Units (iterable + type filter)."""

    @property
    def exists(self):
        return len(self) > 0

    @property
    def first(self):
        return self[0] if self else None

    def __call__(self, type_id):
        return _UnitsList(u for u in self if getattr(u, "type_id", None) == type_id)


def _make_bot(*, player_id=1, own_units, enemy_units, own_structures, enemy_structures):
    bot = MagicMock()
    bot.player_id = player_id
    bot.units = _UnitsList(own_units)
    bot.enemy_units = _UnitsList(enemy_units)
    bot.structures = _UnitsList(own_structures)
    bot.enemy_structures = _UnitsList(enemy_structures)
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    bot.state.upgrades = frozenset()
    bot.game_info = _make_game_info(128, 128, with_grids=True)
    bot.game_info.playable_area = MagicMock()
    bot.game_info.playable_area.x = 0
    bot.game_info.playable_area.y = 0
    bot.game_info.playable_area.width = 128
    bot.game_info.playable_area.height = 128
    bot.game_data = MagicMock()
    bot.enemy_pylon_pos = {1: None, 2: None}
    bot.unit_target_map = {}
    bot.unit_command_map = {}
    by_tag = {
        u.tag: u
        for u in list(own_units)
        + list(enemy_units)
        + list(own_structures)
        + list(enemy_structures)
    }
    bot.all_units.find_by_tag = lambda tag: by_tag.get(tag)
    return bot


def _objective_sum(map_obs: np.ndarray, active_layers) -> float:
    idx = active_layers.index("command_objective")
    return float(map_obs[idx].sum())


@pytest.mark.parametrize(
    "perspective,role,expected_cmd",
    [
        (1, "attack", CommandConstants.ATTACK),
        (1, "defend", CommandConstants.DEFEND),
        (2, "attack", CommandConstants.ATTACK),
        (2, "defend", CommandConstants.DEFEND),
    ],
)
def test_raw_obs_command_target_from_unit_target_map(
    training_conf, perspective, role, expected_cmd
):
    """Both roles and both players must resolve command_target from unit_target_map."""
    objective = Point2((90.0, 40.0) if role == "attack" else (30.0, 30.0))
    unit = create_mock_unit(UnitTypeId.MARINE, Point2((20.0, 20.0)))
    unit.tag = 10 + perspective

    if perspective == 1:
        bot = _make_bot(
            player_id=1,
            own_units=[unit],
            enemy_units=[],
            own_structures=[],
            enemy_structures=[],
        )
    else:
        bot = _make_bot(
            player_id=1,
            own_units=[],
            enemy_units=[unit],
            own_structures=[],
            enemy_structures=[],
        )
    bot.unit_target_map = {unit.tag: objective}
    bot.unit_command_map = {unit.tag: role}

    builder = SquadObservationBuilder(training_conf)
    raw = builder._build_raw_obs(
        bot,
        perspective_player_id=perspective,
        sorted_tags=[unit.tag],
        pool_high_level_command=role,
        anchor_unit_tag=unit.tag,
    )
    assert raw["command_target"] == objective
    assert raw["high_level_command"] == expected_cmd


@pytest.mark.parametrize(
    "perspective,role",
    [(1, "attack"), (2, "attack"), (1, "defend"), (2, "defend")],
)
def test_build_paints_command_objective_for_roles(training_conf, perspective, role):
    """Global map COMMAND_OBJECTIVE must be nonzero for attack and defend (incl. P2)."""
    objective = Point2((100.0, 100.0) if role == "attack" else (40.0, 40.0))
    unit = create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)))
    unit.tag = 42

    if perspective == 1:
        bot = _make_bot(
            player_id=1,
            own_units=[unit],
            enemy_units=[],
            own_structures=[],
            enemy_structures=[],
        )
    else:
        bot = _make_bot(
            player_id=1,
            own_units=[],
            enemy_units=[unit],
            own_structures=[],
            enemy_structures=[],
        )
    bot.unit_target_map = {unit.tag: objective}
    bot.unit_command_map = {unit.tag: role}

    builder = SquadObservationBuilder(training_conf)
    n_g = len(training_conf["active_layers"])
    n_t = len(training_conf["active_tactical_layers"])
    fake_map = np.zeros((n_g, 128, 128), dtype=np.float32)
    # Paint a blob on command_objective so the assertion checks the layer for both owners.
    obj_idx = training_conf["active_layers"].index("command_objective")
    fake_map[obj_idx, 60:70, 60:70] = 1.0
    with patch.object(
        builder._transformer,
        "transform",
        return_value={
            "map": fake_map,
            "command": np.eye(10, dtype=np.float32)[CommandConstants.get_index(role)],
        },
    ), patch.object(
        builder._transformer,
        "transform_tactical",
        return_value=np.zeros((n_t, 32, 32), np.float32),
    ):
        obs = builder.build(
            bot,
            perspective_player_id=perspective,
            squad_unit_tags=[unit.tag],
            pool_high_level_command=role,
            playable_size=(128.0, 128.0),
            anchor_unit_tag=unit.tag,
        )

    assert _objective_sum(obs["map"], training_conf["active_layers"]) > 0.0
    assert obs["command"][CommandConstants.get_index(role)] == 1.0


def test_squad_builder_paints_command_objective_for_flying_tempest(training_conf):
    """Real transformer: flying attacker + unit_target_map Point2 must stamp COMMAND_OBJECTIVE.

    The role test above mocks transform(), so it never checked painting. Air units
    use the same Point2 contract as ground; a blank layer would look like 'idle'.
    """
    nearby = Point2((58.0, 50.0))
    tempest = create_mock_unit(
        UnitTypeId.TEMPEST,
        Point2((50.0, 50.0)),
        is_flying=True,
        is_mechanical=True,
        radius=1.75,
    )
    tempest.tag = 7
    bot = _make_bot(
        player_id=1,
        own_units=[tempest],
        enemy_units=[],
        own_structures=[],
        enemy_structures=[],
    )
    bot.unit_target_map = {tempest.tag: nearby}
    bot.unit_command_map = {tempest.tag: "attack"}

    builder = SquadObservationBuilder(training_conf)
    obs = builder.build(
        bot,
        perspective_player_id=1,
        squad_unit_tags=[tempest.tag],
        pool_high_level_command="attack",
        playable_size=(128.0, 128.0),
        anchor_unit_tag=tempest.tag,
    )
    assert _objective_sum(obs["map"], training_conf["active_layers"]) > 0.0
    t_idx = training_conf["active_tactical_layers"].index("command_objective")
    assert float(obs["anchor_tactical"][t_idx].sum()) > 0.0


def test_squad_builder_keeps_global_objective_when_pylon_outside_fovea(training_conf):
    """Ladder arenas: pylon can sit outside the 32×32 crop; global layer must still paint."""
    far = Point2((100.0, 100.0))
    tempest = create_mock_unit(
        UnitTypeId.TEMPEST,
        Point2((20.0, 20.0)),
        is_flying=True,
        radius=1.75,
    )
    tempest.tag = 8
    bot = _make_bot(
        player_id=1,
        own_units=[tempest],
        enemy_units=[],
        own_structures=[],
        enemy_structures=[],
    )
    bot.unit_target_map = {tempest.tag: far}
    bot.unit_command_map = {tempest.tag: "attack"}

    builder = SquadObservationBuilder(training_conf)
    obs = builder.build(
        bot,
        perspective_player_id=1,
        squad_unit_tags=[tempest.tag],
        pool_high_level_command="attack",
        playable_size=(128.0, 128.0),
        anchor_unit_tag=tempest.tag,
    )
    assert _objective_sum(obs["map"], training_conf["active_layers"]) > 0.0
    t_idx = training_conf["active_tactical_layers"].index("command_objective")
    assert float(obs["anchor_tactical"][t_idx].sum()) == 0.0


def test_attack_fallback_refreshes_enemy_pylon_pos_for_opponent(training_conf):

    """Without unit_target_map, P2 attack must still get objective from enemy pylon."""
    pylon = create_mock_unit(UnitTypeId.PYLON, Point2((80.0, 80.0)))
    pylon.tag = 99
    pylon.is_structure = True
    attacker = create_mock_unit(UnitTypeId.ZERGLING, Point2((10.0, 10.0)))
    attacker.tag = 7

    # Bot is P1; perspective 2 → enemy_structures are bot.structures (P1 pylons).
    bot = _make_bot(
        player_id=1,
        own_units=[],
        enemy_units=[attacker],
        own_structures=[pylon],
        enemy_structures=[],
    )
    bot.unit_target_map = {}
    bot.unit_command_map = {attacker.tag: "attack"}
    assert bot.enemy_pylon_pos[2] is None

    builder = SquadObservationBuilder(training_conf)
    raw = builder._build_raw_obs(
        bot,
        perspective_player_id=2,
        sorted_tags=[attacker.tag],
        pool_high_level_command="attack",
        anchor_unit_tag=attacker.tag,
    )
    assert raw["command_target"] == pylon.position
    assert bot.enemy_pylon_pos[2] == pylon.position


def test_defend_fallback_uses_own_pylon_for_opponent(training_conf):
    """Without unit_target_map, P2 defend must target own (perspective) pylon."""
    pylon = create_mock_unit(UnitTypeId.PYLON, Point2((55.0, 55.0)))
    pylon.tag = 88
    pylon.is_structure = True
    defender = create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)))
    defender.tag = 5

    # Perspective 2: structures = bot.enemy_structures = P2 structures.
    bot = _make_bot(
        player_id=1,
        own_units=[],
        enemy_units=[defender],
        own_structures=[],
        enemy_structures=[pylon],
    )
    bot.unit_target_map = {}
    bot.unit_command_map = {defender.tag: "defend"}

    builder = SquadObservationBuilder(training_conf)
    raw = builder._build_raw_obs(
        bot,
        perspective_player_id=2,
        sorted_tags=[defender.tag],
        pool_high_level_command="defend",
        anchor_unit_tag=defender.tag,
    )
    assert raw["command_target"] == pylon.position


def test_god_mode_raw_obs_splits_armies_by_owner(training_conf):
    """After debug_control_enemy both armies sit in bot.units; obs must still split.

    Alliance-bucket swap (P2 own = enemy_units) left enemy density empty for P1
    and painted the P2 trainee's own army as enemies — no attack objective.
    """
    p1 = create_mock_unit(UnitTypeId.BANSHEE, Point2((20.0, 20.0)), is_flying=True)
    p1.tag = 1
    p1.owner_id = 1
    p2 = create_mock_unit(UnitTypeId.VOIDRAY, Point2((100.0, 100.0)), is_flying=True)
    p2.tag = 2
    p2.owner_id = 2
    p1_pylon = create_mock_unit(UnitTypeId.PYLON, Point2((25.0, 25.0)))
    p1_pylon.tag = 11
    p1_pylon.owner_id = 1
    p1_pylon.is_structure = True
    p2_pylon = create_mock_unit(UnitTypeId.PYLON, Point2((105.0, 105.0)))
    p2_pylon.tag = 22
    p2_pylon.owner_id = 2
    p2_pylon.is_structure = True

    bot = _make_bot(
        player_id=1,
        own_units=[p1, p2],
        enemy_units=[],
        own_structures=[p1_pylon, p2_pylon],
        enemy_structures=[],
    )
    bot.unit_target_map = {}
    bot.unit_command_map = {1: "attack", 2: "attack"}
    builder = SquadObservationBuilder(training_conf)

    raw1 = builder._build_raw_obs(
        bot,
        perspective_player_id=1,
        sorted_tags=[p1.tag],
        pool_high_level_command="attack",
        anchor_unit_tag=p1.tag,
    )
    assert p1 in raw1["units"]
    assert p2 in raw1["enemy_units"]
    assert p1_pylon in raw1["structures"]
    assert p2_pylon in raw1["enemy_structures"]
    assert raw1["command_target"] == p2_pylon.position

    raw2 = builder._build_raw_obs(
        bot,
        perspective_player_id=2,
        sorted_tags=[p2.tag],
        pool_high_level_command="attack",
        anchor_unit_tag=p2.tag,
    )
    assert p2 in raw2["units"]
    assert p1 in raw2["enemy_units"]
    assert p2_pylon in raw2["structures"]
    assert p1_pylon in raw2["enemy_structures"]
    assert raw2["command_target"] == p1_pylon.position


def test_p2_build_does_not_rotate_map_or_xy_norm(training_conf):
    """P2 perspective swaps owners only; maps and x_norm/y_norm stay world-frame."""
    training_conf = {
        **training_conf,
        "active_unit_scalars": ["x_norm", "y_norm"],
    }
    unit = create_mock_unit(UnitTypeId.MARINE, Point2((32.0, 96.0)), owner_id=2)
    unit.tag = 7
    bot = _make_bot(
        player_id=1,
        own_units=[],
        enemy_units=[unit],
        own_structures=[],
        enemy_structures=[],
    )
    builder = SquadObservationBuilder(training_conf)
    n_g = len(training_conf["active_layers"])
    n_t = len(training_conf["active_tactical_layers"])
    fake_map = np.zeros((n_g, 128, 128), dtype=np.float32)
    fake_map[0, 0, 0] = 1.0
    fake_tactical = np.zeros((n_t, 32, 32), dtype=np.float32)
    fake_tactical[0, 0, 0] = 1.0
    with patch.object(
        builder._transformer,
        "transform",
        return_value={
            "map": fake_map,
            "command": np.zeros(10, dtype=np.float32),
        },
    ), patch.object(
        builder._transformer,
        "transform_tactical",
        return_value=fake_tactical,
    ):
        obs = builder.build(
            bot,
            perspective_player_id=2,
            squad_unit_tags=[unit.tag],
            pool_high_level_command="attack",
            playable_size=(128.0, 128.0),
            anchor_unit_tag=unit.tag,
        )

    assert obs["map"][0, 0, 0] == pytest.approx(1.0)
    assert obs["map"][0, -1, -1] == pytest.approx(0.0)
    assert obs["anchor_tactical"][0, 0, 0] == pytest.approx(1.0)
    assert obs["anchor_tactical"][0, -1, -1] == pytest.approx(0.0)
    xi = training_conf["active_unit_scalars"].index("x_norm")
    yi = training_conf["active_unit_scalars"].index("y_norm")
    assert obs["unit_scalars"][0, xi] == pytest.approx(32.0 / 128.0)
    assert obs["unit_scalars"][0, yi] == pytest.approx(96.0 / 128.0)

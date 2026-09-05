"""In-game CTF role chat is posted after debug spawn calls."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from train.envs.ctf.role_chat import format_ctf_role_chat_lines


def test_format_ctf_role_chat_lines_names_policy_and_role():
    lines = format_ctf_role_chat_lines(
        episode=2,
        trainee_race="zerg",
        trainee_player_id=2,
        trainee_roles=["A:attack"],
        opponent_race="terran",
        opponent_player_id=1,
        opponent_roles=["A:defend"],
        opponent_policy="frozen",
    )
    assert lines[0] == "Ep2 TRAINEE: Zerg A:attack (P2)"
    assert lines[1] == "Ep2 OPPONENT: Terran A:defend (P1, frozen)"


def test_env_role_chat_lines_follow_alternate_trainee():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.episode_counter = 2
    env._alternate_trainee_race = True
    env.unit_pools_config = [
        {"race": "zerg", "high_level_command": "attack", "region": "A"},
        {"race": "protoss", "high_level_command": "defend", "region": "A"},
    ]
    env.training_conf = {}
    env.opponent_model = object()
    env.model = None

    lines = env._ctf_role_chat_lines()
    assert "TRAINEE: Protoss A:defend (P2)" in lines[0]
    assert "OPPONENT: Zerg A:attack (P1, frozen)" in lines[1]


def test_send_ctf_role_chat_uses_bot_chat_send():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env._log = MagicMock()
    lines = ["Ep1 TRAINEE: Zerg attack (P1)", "Ep1 OPPONENT: Terran defend (P2, frozen)"]
    env._ctf_role_chat_lines = MagicMock(return_value=lines)
    env.bot = SimpleNamespace(chat_send=AsyncMock())
    asyncio.run(env._send_ctf_role_chat())
    assert env.bot.chat_send.await_count == 2
    env.bot.chat_send.assert_any_await("Ep1 TRAINEE: Zerg attack (P1)", team_only=False)
    env._log.assert_any_call(lines[0])
    env._log.assert_any_call(lines[1])


def test_send_ctf_role_chat_prints_even_without_chat_send():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env._log = MagicMock()
    env._ctf_role_chat_lines = MagicMock(
        return_value=["Ep1 TRAINEE: Zerg A:attack (P1)"]
    )
    env.bot = SimpleNamespace()
    asyncio.run(env._send_ctf_role_chat())
    env._log.assert_any_call("Ep1 TRAINEE: Zerg A:attack (P1)")


def test_spawn_sends_role_chat_after_debug_helpers():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    order = []

    async def _mark(name, result=None):
        async def _fn(*_a, **_k):
            order.append(name)
            return result

        return _fn

    env.unit_pools_config = [{"race": "zerg", "high_level_command": "attack", "region": "A"}]
    env.bot = MagicMock()
    grid = MagicMock()
    grid.width = 8
    grid.height = 8
    grid.__getitem__ = MagicMock(return_value=1)
    env.bot.game_info.pathing_grid = grid
    env.bot.client.step = AsyncMock()
    pylons = MagicMock()
    pylons.exists = False
    env.bot.all_units = MagicMock(return_value=pylons)

    env._clear_ctf_map_units = AsyncMock(side_effect=lambda: order.append("debug"))
    env._spawn_from_unit_pools = AsyncMock(
        side_effect=lambda *_a, **_k: (order.append("debug") or ([], {}, [], []))
    )
    env._refresh_bot_state = AsyncMock()
    env._apply_cloak_env_mode_upgrades = AsyncMock(side_effect=lambda: order.append("debug"))
    env._apply_drop_env_mode_upgrades = AsyncMock()
    env._apply_upgrades_env_mode = AsyncMock(side_effect=lambda: order.append("debug"))
    env._apply_spellcaster_env_mode_upgrades = AsyncMock()
    env._apply_ctf_resources = AsyncMock(side_effect=lambda: order.append("debug"))
    env._apply_heal_map_creep = AsyncMock(side_effect=lambda *_a, **_k: order.append("debug"))
    env._apply_spellcaster_energy = AsyncMock(side_effect=lambda: order.append("debug"))
    env._send_ctf_role_chat = AsyncMock(side_effect=lambda: order.append("chat"))

    asyncio.run(env._spawn_initial_units())

    assert "debug" in order
    assert order[-1] == "chat"
    assert order.index("chat") == len(order) - 1

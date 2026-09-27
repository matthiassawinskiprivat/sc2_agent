"""CTF upgrade curriculum configs and progressive debug_upgrade schedule."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.ctf.combat_upgrade_apply import (
    apply_scheduled_debug_upgrades,
    desired_debug_upgrade_calls,
)
from src.utils.config_manager import ConfigManager


def test_ctf_level_27_29_upgrade_config():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_27", "ctf_level_28", "ctf_level_29"):
        cfg = cm.get_level_config(level)
        modes = {p.strip() for p in str(cfg["env"]["env_mode"]).split(",") if p.strip()}
        # Drop is deferred to pipeline-end transport; upgrades intro has no drop yet.
        assert modes == {"heal", "cloak", "spellcaster", "upgrades"}, level
        assert cfg["env"].get("include_upgrades") is True


def test_levels_after_ctf_27_include_upgrades():
    """The upgrade observation stays on for every level after the upgrades intro."""
    cm = ConfigManager("config_universal")
    start = cm.levels.index("ctf_level_27")
    later = cm.levels[start + 1 :]
    assert later, "expected levels after ctf_level_27"
    for level in later:
        cfg = cm.get_level_config(level)
        assert cfg["env"].get("include_upgrades") is True, level


def test_pipeline_registers_27_29():
    cm = ConfigManager("config_universal")
    assert "ctf_level_27" in cm.levels
    assert "ctf_level_28" in cm.levels
    assert "ctf_level_29" in cm.levels


def test_desired_debug_upgrade_calls_every_k_100():
    k = 100
    assert desired_debug_upgrade_calls(1, k) == 0
    assert desired_debug_upgrade_calls(24, k) == 0
    assert desired_debug_upgrade_calls(25, k) == 1
    assert desired_debug_upgrade_calls(49, k) == 1
    assert desired_debug_upgrade_calls(50, k) == 2
    assert desired_debug_upgrade_calls(74, k) == 2
    assert desired_debug_upgrade_calls(75, k) == 3
    assert desired_debug_upgrade_calls(99, k) == 3
    # Hard-reset boundary → fresh game
    assert desired_debug_upgrade_calls(100, k) == 0
    assert desired_debug_upgrade_calls(125, k) == 1


def test_apply_scheduled_debug_upgrades_only_calls_delta():
    bot = MagicMock()
    bot.client.debug_upgrade = AsyncMock()
    bot.client.step = AsyncMock()
    refresh = AsyncMock()

    applied = asyncio.run(
        apply_scheduled_debug_upgrades(
            bot=bot,
            episode_counter=25,
            every_k=100,
            already_applied=0,
            refresh_state=refresh,
        )
    )
    assert applied == 1
    assert bot.client.debug_upgrade.await_count == 1

    applied = asyncio.run(
        apply_scheduled_debug_upgrades(
            bot=bot,
            episode_counter=50,
            every_k=100,
            already_applied=1,
            refresh_state=refresh,
        )
    )
    assert applied == 2
    assert bot.client.debug_upgrade.await_count == 2  # one more

    # Already at schedule — no extra calls
    applied = asyncio.run(
        apply_scheduled_debug_upgrades(
            bot=bot,
            episode_counter=60,
            every_k=100,
            already_applied=2,
            refresh_state=refresh,
        )
    )
    assert applied == 2
    assert bot.client.debug_upgrade.await_count == 2


def _make_env(env_mode, every_k=100, episode=1):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    conf = {
        "env": {"env_mode": env_mode, "every_k_episodes": every_k},
        "env_mode": env_mode,
        "every_k_episodes": every_k,
    }
    env.training_conf = conf
    env.episode_counter = episode
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.debug_upgrade = AsyncMock()
    env.bot.client.step = AsyncMock()
    env._refresh_bot_state = AsyncMock()
    env._debug_upgrade_tier_applied = 0
    env._upgrade_cache = None
    return env


def test_cloak_debug_upgrade_skipped_when_upgrades_mode():
    env = _make_env("heal,cloak,upgrades")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()


def test_cloak_debug_upgrade_runs_without_upgrades_mode():
    env = _make_env("heal,cloak")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == 20


def test_drop_skips_when_upgrades_mode():
    env = _make_env("heal,drop,upgrades")
    asyncio.run(env._apply_drop_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()


def test_upgrades_mode_calls_debug_upgrade_at_25_percent():
    env = _make_env("heal,cloak,upgrades", every_k=100, episode=25)
    asyncio.run(env._apply_upgrades_env_mode())
    assert env.bot.client.debug_upgrade.await_count == 1
    assert env._debug_upgrade_tier_applied == 1

    # Soft reset same tier — no extra call
    asyncio.run(env._apply_upgrades_env_mode())
    assert env.bot.client.debug_upgrade.await_count == 1

    env.episode_counter = 50
    asyncio.run(env._apply_upgrades_env_mode())
    assert env.bot.client.debug_upgrade.await_count == 2
    assert env._debug_upgrade_tier_applied == 2

    env.episode_counter = 75
    asyncio.run(env._apply_upgrades_env_mode())
    assert env.bot.client.debug_upgrade.await_count == 3
    assert env._debug_upgrade_tier_applied == 3


def test_upgrades_mode_noop_before_25_percent():
    env = _make_env("upgrades", every_k=100, episode=10)
    asyncio.run(env._apply_upgrades_env_mode())
    env.bot.client.debug_upgrade.assert_not_called()
    assert env._debug_upgrade_tier_applied == 0

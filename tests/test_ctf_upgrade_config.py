"""CTF upgrade curriculum configs and env_mode gating."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from src.utils.config_manager import ConfigManager


def test_ctf_level_27_29_upgrade_config():
    cm = ConfigManager("config_universal")
    for level in ("ctf_level_27", "ctf_level_28", "ctf_level_29"):
        cfg = cm.get_level_config(level)
        assert cfg["env"]["env_mode"] == "heal,cloak,drop,upgrades"
        assert cfg["env"].get("include_upgrades") is True


def test_pipeline_registers_27_29():
    cm = ConfigManager("config_universal")
    assert "ctf_level_27" in cm.levels
    assert "ctf_level_28" in cm.levels
    assert "ctf_level_29" in cm.levels


def _make_env(env_mode):
    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    conf = {"env": {"env_mode": env_mode}, "env_mode": env_mode}
    env.training_conf = conf
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.debug_upgrade = AsyncMock()
    env.bot.client.step = AsyncMock()
    env._refresh_bot_state = AsyncMock()
    return env


def test_cloak_debug_upgrade_skipped_when_upgrades_mode():
    env = _make_env("heal,cloak,drop,upgrades")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    env.bot.client.debug_upgrade.assert_not_called()


def test_cloak_debug_upgrade_runs_without_upgrades_mode():
    env = _make_env("heal,cloak")
    asyncio.run(env._apply_cloak_env_mode_upgrades())
    assert env.bot.client.debug_upgrade.await_count == 20

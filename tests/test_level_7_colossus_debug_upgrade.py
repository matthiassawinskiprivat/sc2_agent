import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.level_7_bot import Level7Bot


def test_colossus_debug_upgrade_calls_debug_upgrade_twice():
    bot = object.__new__(Level7Bot)
    bot.client = MagicMock()
    bot.client.debug_upgrade = AsyncMock()
    bot.client.step = AsyncMock()
    bot._refresh_state_local = AsyncMock()

    asyncio.run(bot._debug_unlock_colossus_extended_thermal_lance())

    assert bot.client.debug_upgrade.await_count == 2
    assert bot.client.step.await_count == 2
    assert bot._refresh_state_local.await_count == 2


def test_level7_env_maps_colossus_to_protoss():
    from train.envs.level_7_env import Level7Env
    from sc2.data import Race

    env = object.__new__(Level7Env)
    assert env.get_race_from_unit_type("COLOSSUS") == Race.Protoss

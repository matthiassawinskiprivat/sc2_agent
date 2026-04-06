"""Regression: failed SC2 debug cleanup must signal env to hard-reset SC2."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, PropertyMock, patch
from sc2.bot_ai import BotAI

from src.sc2_bot import SC2Bot, MapCleanupFailedError


class _StuckKillableUnits:
    """Non-empty killable set that never clears (simulates stuck SC2 state)."""

    def filter(self, pred):
        return self

    def __bool__(self):
        return True

    @property
    def amount(self):
        return 2

    def __iter__(self):
        u1, u2 = MagicMock(), MagicMock()
        u1.tag, u2.tag = 1, 2
        return iter([u1, u2])


def test_sc2bot_reset_raises_map_cleanup_failed_when_units_stuck():
    async def run():
        bot = SC2Bot(conn=MagicMock())
        bot.client = AsyncMock()
        bot._refresh_state = AsyncMock()
        stuck = _StuckKillableUnits()
        with patch.object(BotAI, "all_units", new_callable=PropertyMock, return_value=stuck):
            with pytest.raises(MapCleanupFailedError) as exc_info:
                await bot.reset_game_state()
        assert exc_info.value.remaining_count == 2

    asyncio.run(run())


def test_sc2bot_reset_strict_empty_map_false_does_not_raise():
    async def run():
        bot = SC2Bot(conn=MagicMock())
        bot.client = AsyncMock()
        bot._refresh_state = AsyncMock()
        stuck = _StuckKillableUnits()
        with patch.object(BotAI, "all_units", new_callable=PropertyMock, return_value=stuck):
            await bot.reset_game_state(strict_empty_map=False)

    asyncio.run(run())

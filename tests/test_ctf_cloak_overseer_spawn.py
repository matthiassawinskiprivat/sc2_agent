"""Cloak-mode detector spawn uses direct debug_create_unit for all races."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

_HELPER = Path(__file__).resolve().parents[1] / "train" / "envs" / "ctf" / "cloak_detectors.py"
_spec = importlib.util.spec_from_file_location("cloak_detectors", _HELPER)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
spawn_cloak_detectors = _mod.spawn_cloak_detectors


def test_spawn_cloak_detectors_creates_all_races_directly():
    bot = MagicMock()
    bot.client.debug_create_unit = AsyncMock()
    bot.client.step = AsyncMock()
    refresh = AsyncMock()
    cmds = [
        [UnitTypeId.OBSERVER, 1, Point2((1, 1)), 1],
        [UnitTypeId.OVERSEER, 1, Point2((2, 2)), 2],
        [UnitTypeId.RAVEN, 1, Point2((3, 3)), 1],
    ]

    asyncio.run(
        spawn_cloak_detectors(bot=bot, detector_cmds=cmds, refresh_state=refresh)
    )

    assert bot.client.debug_create_unit.await_count == 2
    first = bot.client.debug_create_unit.await_args_list[0].args[0]
    second = bot.client.debug_create_unit.await_args_list[1].args[0]
    assert [c[0] for c in first] == [UnitTypeId.OBSERVER, UnitTypeId.RAVEN]
    assert [c[0] for c in second] == [UnitTypeId.OVERSEER]
    assert all(c[3] == 1 for c in first)
    assert all(c[3] == 2 for c in second)
    bot.client.step.assert_awaited()
    refresh.assert_awaited()


def test_spawn_cloak_detectors_noop_when_empty():
    bot = MagicMock()
    bot.client.debug_create_unit = AsyncMock()
    refresh = AsyncMock()
    asyncio.run(spawn_cloak_detectors(bot=bot, detector_cmds=[], refresh_state=refresh))
    bot.client.debug_create_unit.assert_not_awaited()

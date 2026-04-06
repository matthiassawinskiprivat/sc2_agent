"""Regression: debug UI tactical fovea follows client-selected units."""
from unittest.mock import MagicMock

import pytest

from src.sc2_bot import resolve_tactical_active_unit_for_debug


@pytest.fixture
def bot():
    b = MagicMock()
    b._active_unit_tag = None
    return b


def test_prefers_selected_army_unit(bot):
    u_sel = MagicMock(tag=42)
    bot.units = MagicMock()
    bot.units.selected = MagicMock()
    bot.units.selected.exists = True
    bot.units.selected.first = u_sel
    bot.structures = MagicMock()
    bot.structures.selected = MagicMock()
    bot.structures.selected.exists = False

    active, tag = resolve_tactical_active_unit_for_debug(bot)
    assert active is u_sel
    assert tag == 42


def test_prefers_selected_structure_when_no_army_selection(bot):
    u_army = MagicMock(tag=1)
    bot.units = MagicMock()
    bot.units.selected = MagicMock()
    bot.units.selected.exists = False
    bot.units.exists = True
    bot.units.first = u_army

    s_sel = MagicMock(tag=99)
    bot.structures = MagicMock()
    bot.structures.selected = MagicMock()
    bot.structures.selected.exists = True
    bot.structures.selected.first = s_sel

    active, tag = resolve_tactical_active_unit_for_debug(bot)
    assert active is s_sel
    assert tag == 99


def test_falls_back_to_first_unit_when_nothing_selected(bot):
    u_first = MagicMock(tag=7)
    bot.units = MagicMock()
    bot.units.selected = MagicMock()
    bot.units.selected.exists = False
    bot.units.exists = True
    bot.units.first = u_first
    bot.units.find_by_tag = MagicMock(return_value=None)
    bot.structures = MagicMock()
    bot.structures.selected = MagicMock()
    bot.structures.selected.exists = False
    bot.structures.find_by_tag = MagicMock(return_value=None)

    active, tag = resolve_tactical_active_unit_for_debug(bot)
    assert active is u_first
    assert tag == 7


def test_keeps_last_tag_when_selection_cleared_and_unit_alive(bot):
    u_tracked = MagicMock(tag=3)
    bot._active_unit_tag = 3
    bot.units = MagicMock()
    bot.units.selected = MagicMock()
    bot.units.selected.exists = False
    bot.units.exists = True
    bot.units.first = MagicMock(tag=999)
    bot.units.find_by_tag = MagicMock(return_value=u_tracked)
    bot.structures = MagicMock()
    bot.structures.selected = MagicMock()
    bot.structures.selected.exists = False

    active, tag = resolve_tactical_active_unit_for_debug(bot)
    assert active is u_tracked
    assert tag == 3

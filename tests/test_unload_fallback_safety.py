import pytest
from unittest.mock import MagicMock
import multiprocessing
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from src.sc2_bot import SC2Bot
from tests.unload_test_helpers import create_mock_transport, setup_bot_for_action_test

@pytest.mark.asyncio
async def test_unload_fallback_unknown_unit():
    """TDD: Asserts unknown units use AbilityId.UNLOADALL (no spatial target)."""
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    setup_bot_for_action_test(bot)
    
    # Bunker (not in our spatial mapping)
    bunker = create_mock_transport(UnitTypeId.BUNKER, cargo_used=4, tag=333)
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = bunker
    
    grid_area = 128 * 128
    action = 1 * grid_area + 0
    await bot.apply_action(333, action)
    
    # Should use generic UNLOADALL without target
    bunker.assert_called_with(AbilityId.UNLOADALL)

@pytest.mark.asyncio
async def test_unload_empty_cargo_multi_race():
    """TDD: Asserts no command is issued if cargo is empty for specialized transports."""
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    setup_bot_for_action_test(bot)
    
    medivac = create_mock_transport(UnitTypeId.MEDIVAC, cargo_used=0, tag=444)
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = medivac
    
    grid_area = 128 * 128
    action = 1 * grid_area + 0
    await bot.apply_action(444, action)
    
    assert medivac.call_count == 0

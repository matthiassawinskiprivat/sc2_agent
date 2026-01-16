import pytest
from unittest.mock import MagicMock
import multiprocessing
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from src.sc2_bot import SC2Bot
from tests.unload_test_helpers import create_mock_transport, setup_bot_for_action_test

@pytest.mark.asyncio
async def test_unload_spatial_overlord():
    """TDD: Asserts Overlords use AbilityId.UNLOADALLAT_OVERLORD."""
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    setup_bot_for_action_test(bot)
    
    overlord = create_mock_transport(UnitTypeId.OVERLORD, cargo_used=4, tag=111)
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = overlord
    
    grid_area = 128 * 128
    action = 1 * grid_area + 0
    await bot.apply_action(111, action)
    
    from sc2.position import Point2
    overlord.assert_called_with(AbilityId.UNLOADALLAT_OVERLORD, Point2((10.0, 20.0)))

@pytest.mark.asyncio
async def test_unload_spatial_warp_prism():
    """TDD: Asserts Warp Prisms use AbilityId.UNLOADALLAT_WARPPRISM."""
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    setup_bot_for_action_test(bot)
    
    warp_prism = create_mock_transport(UnitTypeId.WARPPRISM, cargo_used=4, tag=222)
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = warp_prism
    
    grid_area = 128 * 128
    action = 1 * grid_area + 0
    await bot.apply_action(222, action)
    
    from sc2.position import Point2
    warp_prism.assert_called_with(AbilityId.UNLOADALLAT_WARPPRISM, Point2((10.0, 20.0)))

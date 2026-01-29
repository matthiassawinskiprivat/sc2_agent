import pytest
from unittest.mock import MagicMock
import multiprocessing
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from src.sc2_bot import SC2Bot
from tests.unload_test_helpers import create_mock_transport, setup_bot_for_action_test

@pytest.mark.asyncio
async def test_unload_spatial_medivac():
    """
    TDD: Asserts that Medivacs use AbilityId.UNLOADALLAT_MEDIVAC with target_pos.
    """
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    
    # 1. Setup mocks
    target_pos_world = (55.5, 66.6)
    mock_unit_agent = MagicMock()
    mock_unit_agent.convert_spatial_action_to_world.return_value = target_pos_world
    setup_bot_for_action_test(bot, mock_unit_agent)
    
    # 2. Create Medivac mock
    medivac = create_mock_transport(UnitTypeId.MEDIVAC, cargo_used=4, tag=999)
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = medivac
    
    # 3. Trigger 'unload' action (head_idx 1)
    grid_area = 128 * 128
    action = 1 * grid_area + 500 # spatial_idx 500
    
    await bot.apply_action(999, action)
    
    # 4. Assert correctness
    # Expected call: unit(AbilityId.UNLOADALLAT_MEDIVAC, Point2((55.5, 66.6)))
    # Note: bot.apply_action calls unit(AbilityId...) which we mocked in helpers
    
    from sc2.position import Point2
    expected_pos = Point2(target_pos_world)
    
    # Check if the unit was called with the correct Ability and Target
    medivac.assert_called_with(AbilityId.UNLOADALLAT_MEDIVAC, expected_pos)
    bot.do.assert_called_once()

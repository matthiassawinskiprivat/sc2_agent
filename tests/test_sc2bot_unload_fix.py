import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sc2.position import Point2
from src.sc2_bot import SC2Bot
import multiprocessing

@pytest.mark.asyncio
async def test_unload_attribute_error_reproduction():
    """
    Reproduces the AttributeError: 'Unit' object has no attribute 'unload_all_at'
    """
    # Setup mock connection
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    
    # Initialize bot with minimal config
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    bot.grid_size = 128
    
    # Mock agent system and unit agent
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    # Mock conversion from spatial to world: 0 -> (0, 0)
    mock_unit_agent.convert_spatial_action_to_world.return_value = (10.0, 20.0)
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # Mock units list and a specific unit
    mock_unit = MagicMock()
    mock_unit.tag = 123
    mock_unit.cargo_used = 1
    
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = mock_unit
    
    # Action index for 'unload' at spatial index 0
    # head_idx = 1 (unload), spatial_idx = 0
    # action = head_idx * grid_area + spatial_idx
    grid_area = 128 * 128
    action = 1 * grid_area + 0
    
    # Mock bot.do to avoid burnysc2 assertion on UnitCommand type
    bot.do = MagicMock()
    
    # This should no longer raise AttributeError
    from sc2.ids.ability_id import AbilityId
    await bot.apply_action(123, action)
    
    # Verify that the unit was called with UNLOADALL
    mock_unit.assert_called_with(AbilityId.UNLOADALL)
    
    # Verify bot.do was called with the result of the unit call
    bot.do.assert_called_once()

@pytest.mark.asyncio
async def test_unload_with_empty_cargo():
    """
    Verifies that unload is NOT called if cargo is 0.
    """
    mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
    bot = SC2Bot(conn=mock_conn, training_conf={"action_heads": ["move", "unload"]})
    bot.grid_size = 128
    
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    mock_unit_agent.convert_spatial_action_to_world.return_value = (10.0, 20.0)
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    mock_unit = MagicMock()
    mock_unit.tag = 123
    mock_unit.cargo_used = 0 # Empty
    
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = mock_unit
    
    grid_area = 128 * 128
    action = 1 * grid_area + 0
    
    # Should not raise anything because it returns early
    await bot.apply_action(123, action)
    
    # Verify no command was generated (mock_unit was never called as a function for AbilityId)
    assert mock_unit.call_count == 0
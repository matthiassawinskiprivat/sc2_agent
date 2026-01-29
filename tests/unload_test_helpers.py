from unittest.mock import MagicMock
from sc2.ids.unit_typeid import UnitTypeId

def create_mock_transport(unit_type: UnitTypeId, cargo_used: int = 1, tag: int = 123):
    """Helper to create a mock SC2 Unit for transport tests."""
    mock_unit = MagicMock()
    mock_unit.type_id = unit_type
    mock_unit.cargo_used = cargo_used
    mock_unit.tag = tag
    # Mocking is_flying if needed
    mock_unit.is_flying = True
    return mock_unit

def setup_bot_for_action_test(bot, mock_unit_agent=None):
    """Helper to setup a bot with necessary mocks for action testing."""
    if mock_unit_agent is None:
        mock_unit_agent = MagicMock()
        mock_unit_agent.convert_spatial_action_to_world.return_value = (10.0, 20.0)
    
    mock_agent_system = MagicMock()
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    bot.grid_size = 128
    bot.do = MagicMock()
    return mock_unit_agent

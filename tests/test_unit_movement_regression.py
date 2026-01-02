import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from train.envs.level_3_env import Level3Env
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit_command import UnitCommand

@pytest.fixture
def mock_training_conf():
    return {
        "targets": 2,
        "grid_size": 32,
        "max_steps": 100,
        "rewards": {
            "success": 100.0,
            "distance_multiplier": 1.0,
            "step_penalty": -0.1
        },
        "active_layers": []
    }

@pytest.mark.asyncio
@patch("train.envs.level_0_env.ConfigManager")
@patch("train.envs.level_0_env.Level0Env._discover_maps")
async def test_level_3_action_moves_unit(mock_discover_maps, mock_config_manager, mock_training_conf):
    """
    Test that Level3Bot's apply_action actually queues a move command.
    """
    mock_discover_maps.return_value = ["map_01"]
    mock_cm_instance = mock_config_manager.return_value
    mock_cm_instance.get_level_config.return_value = {"env": {"targets": 2, "rewards": mock_training_conf["rewards"]}}
    mock_cm_instance.get_cumulative_active_layers.return_value = []

    env = Level3Env(level_name="level_3", training_conf=mock_training_conf)
    
    # We need a real-ish Level3Bot to test the command queueing logic
    from src.level_3_bot import Level3Bot
    
    # Mock connection for Bot
    class DummyConn:
        def send(self, data): pass
        def recv(self): return "continue", None
        
    bot = Level3Bot(conn=DummyConn(), training_conf=mock_training_conf)
    env._bot = bot
    
    # Mock AgentSystem and spatial conversion
    mock_agent_system = MagicMock()
    mock_unit_agent = MagicMock()
    # Mock a move to (10, 10)
    mock_unit_agent.convert_spatial_action_to_world.return_value = (10, 10) 
    mock_agent_system.unit_agents = {"agent_0": mock_unit_agent}
    bot.agent_system = mock_agent_system
    
    # Mock Units
    unit_tag = 101
    mock_unit = MagicMock()
    mock_unit.tag = unit_tag
    mock_unit.position = Point2((5, 5))
    
    # Create a mock UnitCommand
    mock_cmd = MagicMock(spec=UnitCommand)
    mock_cmd.unit = mock_unit  # Crucial: Link the command back to the unit
    mock_unit.move = MagicMock(return_value=mock_cmd) 
    
    # Mock self.units.find_by_tag
    bot.units = MagicMock()
    bot.units.find_by_tag.return_value = mock_unit
    
    # Initialize bot actions queue if it exists (sc2 bot structure)
    if not hasattr(bot, "_actions"):
        bot._actions = []
        
    # Mock env client
    env._client = MagicMock()
    env._client.step = AsyncMock()
    env._client.actions = AsyncMock()
    env._client.observation = AsyncMock()
    
    # Set active unit tags for the env
    env.active_unit_tags = [unit_tag]
    env.current_unit_index = 0
    
    # Mock bot.get_active_zerglings to return our mock unit (needed for loop advancement)
    bot.get_active_zerglings = MagicMock(return_value=[mock_unit])
    
    # Mock bot calculations to avoid errors
    bot.calculate_distance_reward = MagicMock(return_value=0)
    bot.check_collections = AsyncMock(return_value={})
    bot.get_observation_for_unit = MagicMock(return_value=(np.zeros((1, 32, 32)), {}))
    
    # Mock game_info for ProtoWrapper
    bot.game_info = MagicMock()
    bot.game_info._proto = MagicMock()
    
    # Mock GameState and ProtoWrapper for _prepare_step
    with patch("sc2.game_state.GameState"), \
         patch("train.envs.level_0_env.ProtoWrapper"):
        bot._prepare_step = MagicMock()
        
        # --- ACT ---
        # Call env._async_step instead of bot.apply_action
        await env._async_step(action=123)

    # --- ASSERT ---
    # 1. Verify spatial conversion was called
    mock_unit_agent.convert_spatial_action_to_world.assert_called_with(123)
    
    # 2. Verify unit.move() was called with target (10, 10)
    mock_unit.move.assert_called_with(Point2((10, 10)))
    
    # 3. Verify env._client.actions was called with the mock command!
    # This confirms the environment actually transmitted the queued action.
    env._client.actions.assert_called()
    
    # Check args
    args, _ = env._client.actions.call_args
    sent_actions = args[0]
    assert len(sent_actions) > 0
    assert sent_actions[0] == mock_cmd
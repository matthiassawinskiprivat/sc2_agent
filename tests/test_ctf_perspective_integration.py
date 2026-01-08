import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from src.agent.ctf_bot import CaptureTheFlagBot
from src.utils.perspective_transformer import PerspectiveTransformer
from sc2.position import Point2

@pytest.mark.asyncio
async def test_ctf_perspective_integration():
    # We want to verify that when player_id is 2:
    # 1. get_observation returns a flipped array
    # 2. apply_action flips the action index before using it
    # 3. action_masks returns a flipped mask
    
    training_conf = {
        "grid_size": 4,
        "env": {"active_layers_count": 1}
    }
    
    bot = CaptureTheFlagBot(player_id=2, training_conf=training_conf)
    await bot.on_start()
    
    # Mock game_info and game_data after on_start replaced them with real system if it could
    bot.game_info = MagicMock()
    bot.game_info.map_size = MagicMock(width=100, height=100)
    bot.game_info.pathing_grid = MagicMock(width=100, height=100)
    bot.game_data = MagicMock()
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    
    # Mock agent system and unit agent
    bot.agent_system = MagicMock()
    unit_agent = MagicMock()
    bot.agent_system.unit_agents = {"active_agent": unit_agent}
    
    # Mock a processed observation (1 layer, 4x4)
    # [1, 0, 0, 0,
    #  0, 0, 0, 0,
    #  0, 0, 0, 0,
    #  0, 0, 0, 2]
    mock_obs = np.zeros((1, 4, 4), dtype=np.uint8)
    mock_obs[0, 0, 0] = 1
    mock_obs[0, 3, 3] = 2
    bot.agent_system.process_observation.return_value = {"active_agent": mock_obs}
    
    # Verify Observation Flipping
    obs = bot.get_observation()
    # Should be flipped
    assert obs[0, 3, 3] == 1
    assert obs[0, 0, 0] == 2
    
    # Verify Action Mask Flipping
    # [True, False, False, False, ...] (idx 0 is True)
    mock_mask = np.zeros(16, dtype=bool)
    mock_mask[0] = True
    unit_agent.get_action_mask.return_value = mock_mask
    
    mask = bot.action_masks()
    # Since it's dual head (Move + Attack), size is 32
    # Move half should be flipped: idx 15 should be True
    assert mask[15] == True
    assert mask[0] == False
    
    # Verify Action Flipping in apply_action
    # If agent outputs Move to (0,0) [idx 0], it should be flipped to (3,3) [idx 15] for P2
    bot.all_units = MagicMock()
    unit = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    unit.owner_id = 2
    
    # We need to mock convert_spatial_action_to_world to see what it receives
    unit_agent.convert_spatial_action_to_world.side_effect = lambda idx: (idx, idx)
    
    bot.do = MagicMock()
    await bot.apply_action(123, 0)
    
    # Check that convert_spatial_action_to_world was called with flipped index 15
    unit_agent.convert_spatial_action_to_world.assert_called_with(15)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ctf_perspective_integration())

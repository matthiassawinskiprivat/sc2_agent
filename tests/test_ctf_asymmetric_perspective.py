import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from src.agent.ctf_bot import CaptureTheFlagBot
from src.utils.perspective_transformer import PerspectiveTransformer

@pytest.mark.asyncio
async def test_bot_automated_perspective_consistency():
    """Verify that automated Player 2 uses flipped perspectives correctly."""
    grid_size = 128
    bot2 = CaptureTheFlagBot(player_id=2, training_conf={"grid_size": grid_size})
    bot2.is_trainee = False
    
    # Mock model
    mock_model = MagicMock()
    # Predict returns action 10 (bottom-left in flipped space)
    mock_model.predict.return_value = (np.array(10), None)
    bot2.model = mock_model
    
    # Mock observation and mask methods to return identifiable data
    # Normal obs: [1, 0, 0] at top-left
    # Flipped obs: should be [1, 0, 0] at bottom-right
    raw_obs = np.zeros((23, grid_size, grid_size))
    raw_obs[0, 0, 0] = 1.0
    
    bot2.get_observation = MagicMock(return_value=PerspectiveTransformer.flip_observation(raw_obs))
    bot2.action_masks = MagicMock(return_value=np.ones(2 * grid_size * grid_size, dtype=bool))
    
    action = bot2.predict_action()
    assert action == 10
    
    # Now verify apply_action flips it back
    bot2.all_units = MagicMock()
    mock_unit = MagicMock()
    mock_unit.owner_id = 2
    bot2.all_units.find_by_tag.return_value = mock_unit
    bot2.do = MagicMock()
    bot2.agent_system = MagicMock()
    bot2.agent_system.unit_agents = {"active_agent": MagicMock()}
    bot2.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.return_value = (5, 5)
    
    await bot2.apply_action(201, action)
    
    # Verify that PerspectiveTransformer.flip_action_idx was called internally or logic matches
    # Action 10 for Player 2 should have been transformed
    # We can check if the internal world conversion received the flipped index
    expected_flipped_action = PerspectiveTransformer.flip_action_idx(10, grid_size)
    
    args, _ = bot2.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.call_args
    assert args[0] == expected_flipped_action

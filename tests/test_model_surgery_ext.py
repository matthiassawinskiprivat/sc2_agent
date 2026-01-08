import pytest
import torch
from src.utils.model_utils import graft_model_weights

def test_graft_action_expansion_level0_to_drop():
    """Test expanding actions from 1N to 4N (Level0 -> Drop)."""
    grid_area = 100
    old_dim = 1 * grid_area # Move
    new_dim = 4 * grid_area # Move + Attack + Load + Unload
    in_features = 10
    
    state_dict = {
        "action_net.weight": torch.randn(old_dim, in_features),
        "action_net.bias": torch.randn(old_dim)
    }
    
    new_sd = graft_model_weights(state_dict, new_in_channels=5, new_action_dim=new_dim, action_handling_mode='expand')
    
    if "action_net.weight" not in new_sd:
        pytest.fail("action_net.weight was dropped instead of grafted!")
        
    new_w = new_sd["action_net.weight"]
    assert new_w.shape == (new_dim, in_features)
    assert torch.equal(new_w[:old_dim], state_dict["action_net.weight"])
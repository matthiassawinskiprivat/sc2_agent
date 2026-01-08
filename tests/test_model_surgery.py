import pytest
import torch
import copy
from src.utils.model_utils import graft_model_weights

def test_graft_action_expansion_2x():
    """Test expanding actions from N to 2N (e.g. Move -> Move+Attack)."""
    grid_area = 100
    old_dim = grid_area
    new_dim = 2 * grid_area
    in_features = 10
    
    state_dict = {
        "action_net.weight": torch.ones(old_dim, in_features), # All ones
        "action_net.bias": torch.ones(old_dim)
    }
    
    new_sd = graft_model_weights(state_dict, new_in_channels=5, new_action_dim=new_dim)
    
    # Check Weight Shape
    new_w = new_sd["action_net.weight"]
    assert new_w.shape == (new_dim, in_features)
    
    # Check First Half (Should be preserved/copied)
    # The current logic preserves 'Move' in the first half.
    assert torch.equal(new_w[:old_dim], state_dict["action_net.weight"])
    
    # Check Second Half (Should be random/initialized)
    # It shouldn't be all ones
    assert not torch.equal(new_w[old_dim:], torch.ones(old_dim, in_features))

def test_graft_action_expansion_level7_to_drop():
    """Test expanding actions from 2N to 4N (Level7 -> Drop)."""
    grid_area = 100
    old_dim = 2 * grid_area # Move + Attack
    new_dim = 4 * grid_area # Move + Attack + Load + Unload
    in_features = 10
    
    state_dict = {
        "action_net.weight": torch.randn(old_dim, in_features),
        "action_net.bias": torch.randn(old_dim)
    }
    
    new_sd = graft_model_weights(state_dict, new_in_channels=5, new_action_dim=new_dim)
    
    new_w = new_sd["action_net.weight"]
    
    # Verify shape
    assert new_w.shape == (new_dim, in_features)
    
    # Verify preservation of Move+Attack (First 2 blocks)
    assert torch.equal(new_w[:old_dim], state_dict["action_net.weight"])

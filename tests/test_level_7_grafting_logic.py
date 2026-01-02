import torch
import pytest
from src.utils.model_utils import graft_model_weights

def test_graft_action_head_expansion_preservation():
    # 1. Setup a mock state_dict with a square action head
    # old_grid = 4, old_dim = 16
    old_grid = 4
    old_dim = old_grid * old_grid
    in_features = 8
    
    # Weights: (old_dim, in_features)
    action_weight = torch.ones(old_dim, in_features) * 5.0
    action_bias = torch.ones(old_dim) * 2.0
    
    state_dict = {
        "action_net.weight": action_weight.clone(),
        "action_net.bias": action_bias.clone()
    }
    
    # 2. Target: new_action_dim = 2 * old_dim = 32
    # This is NOT a square (sqrt(32) is not int), 
    # but our new logic should handle the "2x" case specifically.
    new_action_dim = 2 * old_dim
    
    # Perform grafting
    new_state_dict = graft_model_weights(state_dict, new_in_channels=3, new_action_dim=new_action_dim)
    
    # 3. Verify
    new_weight = new_state_dict["action_net.weight"]
    new_bias = new_state_dict["action_net.bias"]
    
    assert new_weight.shape == (new_action_dim, in_features)
    assert new_bias.shape == (new_action_dim,)
    
    # First half should match old weights exactly
    assert torch.allclose(new_weight[:old_dim, :], action_weight)
    assert torch.allclose(new_bias[:old_dim], action_bias)
    
    # Second half should be different (randomized/initialized)
    # We check that it's not all 5.0 (unlikely for random noise)
    assert not torch.allclose(new_weight[old_dim:, :], action_weight)

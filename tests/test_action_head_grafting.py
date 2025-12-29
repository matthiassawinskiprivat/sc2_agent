import torch
import pytest
from src.utils.model_utils import graft_model_weights

def test_graft_action_heads_upscale():
    """
    Test upscaling action heads from 32x32 (1024) to 64x64 (4096).
    """
    old_grid = 32
    new_grid = 64
    old_dim = old_grid * old_grid
    new_dim = new_grid * new_grid
    features = 512
    
    state_dict = {
        "features_extractor.cnn.0.weight": torch.randn(32, 4, 8, 8), # Input graft (unchanged here)
        "action_net.weight": torch.randn(old_dim, features),
        "action_net.bias": torch.randn(old_dim),
        "value_net.weight": torch.randn(1, features), # Scalar value, should be kept
        "value_net.bias": torch.randn(1)
    }
    
    # We pass new_in_channels=4 (no change) and new_action_dim=4096
    new_state_dict = graft_model_weights(state_dict, new_in_channels=4, new_action_dim=new_dim)
    
    # Check action_net.weight
    assert new_state_dict["action_net.weight"].shape == (new_dim, features)
    assert new_state_dict["action_net.bias"].shape == (new_dim,)
    
    # Check value_net (should be untouched)
    assert new_state_dict["value_net.weight"].shape == (1, features)
    
    # Ensure it's not just random noise (simple check: mean roughly preserved)
    # The bias is 1D, so it goes through reshape->interpolate->flatten
    assert abs(new_state_dict["action_net.bias"].mean() - state_dict["action_net.bias"].mean()) < 0.5

def test_graft_action_heads_mismatch_delete():
    """
    Test that if dimensions don't look like a spatial grid, the key is deleted
    to avoid shape mismatch errors during loading.
    """
    old_dim = 101 # Not a square
    new_dim = 4096
    features = 512
    
    state_dict = {
        "action_net.weight": torch.randn(old_dim, features),
        "action_net.bias": torch.randn(old_dim),
    }
    
    new_state_dict = graft_model_weights(state_dict, new_in_channels=4, new_action_dim=new_dim)
    
    assert "action_net.weight" not in new_state_dict
    assert "action_net.bias" not in new_state_dict

def test_graft_input_and_output_simultaneously():
    """
    Test changing input channels AND action dim at the same time.
    """
    old_grid = 32
    new_grid = 64
    old_dim = old_grid * old_grid
    new_dim = new_grid * new_grid
    
    state_dict = {
        "features_extractor.cnn.0.weight": torch.randn(32, 5, 8, 8), # 5 channels
        "action_net.weight": torch.randn(old_dim, 512)
    }
    
    # Change input to 4 channels, output to 64x64
    new_state_dict = graft_model_weights(state_dict, new_in_channels=4, new_action_dim=new_dim)
    
    assert new_state_dict["features_extractor.cnn.0.weight"].shape == (32, 4, 8, 8)
    assert new_state_dict["action_net.weight"].shape == (new_dim, 512)

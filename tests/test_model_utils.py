import pytest
import torch
import numpy as np
from src.utils.model_utils import graft_model_weights

def test_graft_model_weights_expands_channels():
    # 1. Setup Dummy State Dict
    # Mimic SB3 CnnPolicy structure
    # Conv Layer 0: 32 filters, 4 input channels, 3x3 kernel
    old_weight = torch.randn(32, 4, 3, 3)
    old_bias = torch.randn(32)
    other_weight = torch.randn(10, 10)
    
    state_dict = {
        "features_extractor.cnn.0.weight": old_weight,
        "features_extractor.cnn.0.bias": old_bias,
        "mlp_extractor.policy_net.0.weight": other_weight
    }
    
    # 2. Execute Surgery
    # Expand input channels from 4 to 6
    new_state_dict = graft_model_weights(state_dict, new_in_channels=6)
    
    # 3. Verify Dimensions
    new_weight = new_state_dict["features_extractor.cnn.0.weight"]
    assert new_weight.shape == (32, 6, 3, 3)
    
    # 4. Verify Preservation of Old Weights
    # The first 4 channels should be identical to the original
    assert torch.equal(new_weight[:, :4, :, :], old_weight)
    
    # 5. Verify Initialization of New Weights
    # The new channels (4 and 5) should differ from zero (random noise) 
    # and should ideally be small.
    new_channels = new_weight[:, 4:, :, :]
    assert not torch.equal(new_channels, torch.zeros_like(new_channels))
    
    # 6. Verify Other Weights Unchanged
    assert torch.equal(new_state_dict["features_extractor.cnn.0.bias"], old_bias)
    assert torch.equal(new_state_dict["mlp_extractor.policy_net.0.weight"], other_weight)

def test_graft_model_weights_no_change_needed():
    # If channels match, do nothing
    old_weight = torch.randn(32, 4, 3, 3)
    state_dict = {
        "features_extractor.cnn.0.weight": old_weight,
    }
    
    new_state_dict = graft_model_weights(state_dict, new_in_channels=4)
    assert new_state_dict is state_dict # Should return same object or identical copy
    assert torch.equal(new_state_dict["features_extractor.cnn.0.weight"], old_weight)

def test_graft_model_weights_shrinks_channels():
    # Edge case: What if we remove layers? (e.g. debugging)
    # The spec focuses on expansion, but the utility should probably handle shrinking or throw error.
    # For now, let's assume we just slice.
    old_weight = torch.randn(32, 4, 3, 3)
    state_dict = {
        "features_extractor.cnn.0.weight": old_weight,
    }
    
    new_state_dict = graft_model_weights(state_dict, new_in_channels=2)
    new_weight = new_state_dict["features_extractor.cnn.0.weight"]
    
    assert new_weight.shape == (32, 2, 3, 3)
    assert torch.equal(new_weight, old_weight[:, :2, :, :])

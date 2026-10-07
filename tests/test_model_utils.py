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
    # Since graft_model_weights performs a deepcopy, 'is' will be false.
    # We check that the weights are identical.
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


def test_graft_model_weights_new_action_bias_is_neutral():
    """Newly added action logits should start neutral, not strongly biased."""
    old_bias = torch.tensor([0.2, -0.1], dtype=torch.float32)
    state_dict = {"action_net.bias": old_bias}
    target_state = {"action_net.bias": torch.zeros(4, dtype=torch.float32)}

    new_state_dict = graft_model_weights(
        state_dict,
        new_in_channels=1,
        new_action_dim=4,
        target_state_dict=target_state,
    )

    new_bias = new_state_dict["action_net.bias"]
    assert torch.allclose(new_bias[:2], old_bias)
    assert torch.allclose(new_bias[2:], torch.zeros(2))


def test_graft_model_weights_zero_fills_new_unit_scalar_column():
    """A trailing unit scalar column must not disturb old columns."""
    old = torch.arange(8 * 48, dtype=torch.float32).reshape(8, 48)
    keys = (
        "features_extractor.entity_encoder.encoder.0.weight",
        "features_extractor.enemy_encoder.encoder.0.weight",
    )
    state_dict = {key: old.clone() for key in keys}
    target_state = {key: torch.randn(8, 49) for key in keys}

    grafted = graft_model_weights(
        state_dict,
        new_in_channels=1,
        target_state_dict=target_state,
    )

    for key in keys:
        weight = grafted[key]
        assert weight.shape == (8, 49)
        assert torch.equal(weight[:, :48], old)
        assert torch.equal(weight[:, 48], torch.zeros(8))

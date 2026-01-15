import torch
import pytest
from src.utils.model_utils import graft_model_weights

def test_graft_ctf_to_level_drop():
    """
    Verifies that grafting from CTF (33 channels, 32768 actions)
    to LevelDrop (35 channels, 65536 actions) works correctly.
    """
    # 1. Mock CTF state_dict
    # Input channels: 33, features_dim: 512, actions: 32768
    state_dict = {
        "features_extractor.cnn.0.weight": torch.randn(32, 33, 8, 8),
        "pi_features_extractor.cnn.0.weight": torch.randn(32, 33, 8, 8),
        "vf_features_extractor.cnn.0.weight": torch.randn(32, 33, 8, 8),
        "action_net.weight": torch.randn(32768, 512),
        "action_net.bias": torch.randn(32768),
        "value_net.weight": torch.randn(1, 512),
        "value_net.bias": torch.randn(1)
    }

    # 2. Perform Grafting
    new_in_channels = 35
    new_action_dim = 65536
    
    new_state_dict = graft_model_weights(
        state_dict,
        new_in_channels=new_in_channels,
        new_action_dim=new_action_dim
    )

    # 3. Verify Results
    # CNN Weights should be expanded to 35 channels
    assert new_state_dict["features_extractor.cnn.0.weight"].shape == (32, 35, 8, 8)
    
    # Action net should be expanded (32768 * 2 = 65536)
    # This triggers the 'use_expansion' logic in graft_model_weights
    assert new_state_dict["action_net.weight"].shape == (65536, 512)
    assert new_state_dict["action_net.bias"].shape == (65536,)
    
    # Value net should remain unchanged (1, 512)
    assert new_state_dict["value_net.weight"].shape == (1, 512)

    # Verify that data was preserved in the first block
    assert torch.allclose(new_state_dict["action_net.weight"][:32768, :], state_dict["action_net.weight"])

if __name__ == "__main__":
    test_graft_ctf_to_level_drop()

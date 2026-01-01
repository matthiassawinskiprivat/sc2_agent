import torch
import torch.nn as nn
from src.utils.model_utils import graft_model_weights

def test_graft_active_layers_increase():
    # Simulate a model with 4 input channels (Level 0-2)
    # Shape: (out_channels, in_channels, k_h, k_w)
    in_channels_old = 4
    out_channels = 32
    kernel_size = 3
    
    state_dict = {
        "features_extractor.cnn.0.weight": torch.ones(out_channels, in_channels_old, kernel_size, kernel_size) * 0.5,
        "features_extractor.cnn.0.bias": torch.zeros(out_channels),
        "action_net.weight": torch.ones(1024, 512), # Assuming some other layers
        "action_net.bias": torch.zeros(1024)
    }
    
    # Target: 5 input channels (Level 3 adding command_assignment)
    in_channels_new = 5
    
    new_state_dict = graft_model_weights(state_dict, new_in_channels=in_channels_new)
    
    new_weight = new_state_dict["features_extractor.cnn.0.weight"]
    
    assert new_weight.shape == (out_channels, in_channels_new, kernel_size, kernel_size)
    
    # Check if old weights are preserved in the first 4 channels
    assert torch.allclose(new_weight[:, :in_channels_old, :, :], state_dict["features_extractor.cnn.0.weight"])
    
    # Check if new channel is initialized with something else (should be small noise in graft_model_weights)
    # The implementation uses torch.randn(...) * 0.01
    assert not torch.allclose(new_weight[:, in_channels_old:, :, :], torch.ones(out_channels, 1, kernel_size, kernel_size) * 0.5)
    
    print("Test passed: Weights preserved and shape updated correctly.")

if __name__ == "__main__":
    test_graft_active_layers_increase()

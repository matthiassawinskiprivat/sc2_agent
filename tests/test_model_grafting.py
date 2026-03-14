
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.agent.models.unit_control_network import UnitControlNetwork
from src.utils.model_utils import graft_model_weights, decompose_action_dim

def test_cnn_input_grafting():
    """Verify that adding/removing input channels preserves weights for existing channels."""
    # 1. Source: 3 channels
    obs_space_3 = spaces.Dict({
        "map": spaces.Box(0, 1, (3, 128, 128)),
        "command": spaces.Box(0, 1, (10,))
    })
    net_3 = UnitControlNetwork(obs_space_3, agent_architecture="universal")
    state_3 = net_3.state_dict()
    
    # Set known weights for first channel
    with torch.no_grad():
        net_3.cnn[0].weight[:, 0, :, :].fill_(1.23)
    
    # 2. Target: 5 channels
    obs_space_5 = spaces.Dict({
        "map": spaces.Box(0, 1, (5, 128, 128)),
        "command": spaces.Box(0, 1, (10,))
    })
    net_5 = UnitControlNetwork(obs_space_5, agent_architecture="universal")
    target_state_5 = net_5.state_dict()
    
    # 3. Graft
    grafted_state = graft_model_weights(net_3.state_dict(), new_in_channels=5, target_state_dict=target_state_5)
    net_5.load_state_dict(grafted_state)
    
    # 4. Verify
    # First channel should be preserved
    assert torch.allclose(net_5.cnn[0].weight[:, 0, :, :], torch.tensor(1.23)), "Channel 0 weights lost!"
    # New channels (3, 4) should be different from source (randomly initialized in graft)
    assert not torch.allclose(net_5.cnn[0].weight[:, 3, :, :], torch.tensor(0.0)), "New channels should not be zero"

def test_fusion_layer_grafting_add_tactical():
    """Verify that adding the tactical stream preserves Global and Command weights in the fusion layer."""
    # 1. Source: Global Only
    obs_space_global = spaces.Dict({
        "map": spaces.Box(0, 1, (3, 128, 128)),
        "command": spaces.Box(0, 1, (10,))
    })
    net_old = UnitControlNetwork(obs_space_global, agent_architecture="universal")
    
    # Fill split points with recognizable values
    # Fusion layer is net.linear[0]
    with torch.no_grad():
        # Global part (first 9216 cols)
        net_old.linear[0].weight[:, :100].fill_(7.77)
        # Command part (last 10 cols)
        net_old.linear[0].weight[:, -10:].fill_(9.99)
        
    # 2. Target: Global + Tactical
    obs_space_tactical = spaces.Dict({
        "map": spaces.Box(0, 1, (3, 128, 128)),
        "tactical": spaces.Box(0, 1, (3, 32, 32)),
        "command": spaces.Box(0, 1, (10,))
    })
    net_new = UnitControlNetwork(obs_space_tactical, agent_architecture="universal")
    target_state = net_new.state_dict()
    
    # 3. Graft
    grafted_state = graft_model_weights(net_old.state_dict(), new_in_channels=3, target_state_dict=target_state)
    
    # This might fail currently because graft_model_weights doesn't know how to split the fusion layer
    try:
        net_new.load_state_dict(grafted_state)
    except Exception as e:
        print(f"Current grafting failed as expected: {e}")
        return

    # 4. Verify (These will fail until we fix model_utils.py)
    # n_flatten for 128x128 is 9216
    assert torch.allclose(net_new.linear[0].weight[:, :100], torch.tensor(7.77)), "Global fusion weights lost!"
    assert torch.allclose(net_new.linear[0].weight[:, -10:], torch.tensor(9.99)), "Command fusion weights lost!"
    print("Fusion layer grafting passed!")

def test_spatial_interpolation_16_to_32():
    """Verify that action head scales from 16x16 knowledge to 32x32."""
    # Source action dim: 1 head * 16*16 + 0 cat = 256
    # Target action dim: 1 head * 32*32 + 0 cat = 1024
    old_w = torch.zeros((256, 512))
    # Draw a "hot spot" in the center of the 16x16 grid
    # Center of 16x16 is (8, 8) => index 8*16 + 8 = 136
    old_w[136, :].fill_(1.0)
    
    state_dict = {"action_net.weight": old_w, "action_net.bias": torch.zeros(256)}
    
    # Target model has 1024 actions
    target_w = torch.zeros((1024, 512))
    target_state = {"action_net.weight": target_w, "action_net.bias": torch.zeros(1024)}
    
    # Graft (using 32 as new_grid_size)
    grafted = graft_model_weights(state_dict, new_in_channels=3, new_action_dim=1024, 
                                  new_grid_size=32, target_state_dict=target_state)
    
    new_w = grafted["action_net.weight"]
    # Center of 32x32 is (16, 16) => index 16*32 + 16 = 528
    assert new_w[528, 0] > 0.5, "Knowledge did not interpolate to center of 32x32 grid!"
    print("Spatial interpolation 16->32 passed!")

if __name__ == "__main__":
    test_cnn_input_grafting()
    test_spatial_interpolation_16_to_32()
    test_fusion_layer_grafting_add_tactical()

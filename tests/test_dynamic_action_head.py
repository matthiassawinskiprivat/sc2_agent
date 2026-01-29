import pytest
import torch
import torch.nn as nn
from src.models.components.dynamic_action_head import DynamicActionHead

def test_dynamic_action_head_initialization():
    """Test that heads are created based on the list of names."""
    input_dim = 64
    heads = ["move", "attack", "load", "unload"]
    grid_size = 10 * 10
    
    dynamic_head = DynamicActionHead(input_dim, heads, grid_size=grid_size)
    
    # Check Selector
    assert isinstance(dynamic_head.selector, nn.Linear)
    assert dynamic_head.selector.out_features == 4
    
    # Check Spatial Heads
    assert len(dynamic_head.spatial_heads) == 4
    assert "move" in dynamic_head.spatial_heads
    assert "unload" in dynamic_head.spatial_heads
    assert isinstance(dynamic_head.spatial_heads["move"], nn.Linear)
    assert dynamic_head.spatial_heads["move"].out_features == grid_size

def test_dynamic_action_head_forward():
    """Test the forward pass returns correct shapes."""
    input_dim = 32
    batch_size = 2
    heads = ["move", "attack"]
    grid_size = 16
    
    dynamic_head = DynamicActionHead(input_dim, heads, grid_size=grid_size)
    
    dummy_input = torch.randn(batch_size, input_dim)
    
    selector_logits, spatial_outputs = dynamic_head(dummy_input)
    
    # Check Selector Shape [Batch, N_Heads]
    assert selector_logits.shape == (batch_size, 2)
    
    # Check Spatial Outputs
    assert len(spatial_outputs) == 2
    assert spatial_outputs["move"].shape == (batch_size, grid_size)
    assert spatial_outputs["attack"].shape == (batch_size, grid_size)

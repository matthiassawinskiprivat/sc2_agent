import torch
import numpy as np
import pytest
from src.utils.model_utils import interpolate_spatial_weights

def test_interpolate_spatial_weights_upscale():
    """
    Test upscaling weights from 32x32 to 64x64.
    The goal is to ensure spatial hotspots are preserved and mean/variance remain stable.
    """
    # Create a 32x32 weight tensor (e.g., for 1 output channel and 512 input features)
    # Shape: (out_features, in_features)
    # For Discrete(1024), out_features is 1024, which is 32x32 spatially.
    old_size = 32
    new_size = 64
    in_features = 512
    
    # Create a "hotspot" at (10, 10) in the 32x32 grid
    # Flattened index for (10, 10) is 10 * 32 + 10 = 330
    weights = torch.randn(old_size * old_size, in_features) * 0.1
    weights[330, :] += 5.0 # Strong hotspot
    
    original_mean = weights.mean().item()
    original_std = weights.std().item()
    
    # Perform interpolation
    new_weights = interpolate_spatial_weights(weights, old_size, new_size)
    
    # Check shape
    assert new_weights.shape == (new_size * new_size, in_features)
    
    # Check that the hotspot moved to roughly (20, 20) in 64x64 grid
    # Flattened index for (20, 20) is 20 * 64 + 20 = 1300
    # We check a small neighborhood around 1300
    neighborhood = new_weights[1300-2:1300+2, :]
    assert neighborhood.max() > 2.0 # Hotspot should be preserved
    
    # Check statistics (should be roughly preserved)
    # Allowing some margin due to interpolation smoothing
    assert abs(new_weights.mean().item() - original_mean) < 0.1
    assert abs(new_weights.std().item() - original_std) < 0.2

def test_interpolate_spatial_weights_invalid_shape():
    """Test that it raises error for non-matching flattened size."""
    weights = torch.randn(100, 512)
    with pytest.raises(ValueError):
        interpolate_spatial_weights(weights, 32, 64)

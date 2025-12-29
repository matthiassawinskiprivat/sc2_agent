import torch
import torch.nn.functional as F
import copy
from typing import Dict, Any
from loguru import logger

def interpolate_spatial_weights(weights: torch.Tensor, old_size: int, new_size: int) -> torch.Tensor:
    """
    Interpolates spatial weights (e.g., action head weights) from old_size to new_size.
    Assumes weights are flattened spatially in the first dimension: (old_size*old_size, in_features)
    
    Args:
        weights: Tensor of shape (old_size*old_size, in_features)
        old_size: The original spatial dimension (e.g., 32)
        new_size: The target spatial dimension (e.g., 64)
        
    Returns:
        Interpolated tensor of shape (new_size*new_size, in_features)
    """
    if weights.shape[0] != old_size * old_size:
        raise ValueError(f"Weights first dimension {weights.shape[0]} does not match spatial size {old_size}x{old_size}")

    # weights: (spatial, features) -> (features, spatial)
    # Then reshape to (1, features, old_size, old_size) for 2D interpolation
    in_features = weights.shape[1]
    x = weights.t().view(1, in_features, old_size, old_size)
    
    # Perform bilinear interpolation
    # align_corners=True ensures the edges match correctly
    x_new = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=True)
    
    # Reshape back to (new_size*new_size, in_features)
    # (1, features, new_size, new_size) -> (features, new_size*new_size) -> (new_size*new_size, features)
    new_weights = x_new.view(in_features, new_size * new_size).t()
    
    # Preservation of statistics: Bilinear interpolation tends to smooth values, 
    # which can reduce variance. We rescale to match original standard deviation.
    # Note: Mean is generally preserved by bilinear interpolation.
    
    old_std = weights.std()
    new_std = new_weights.std()
    
    if new_std > 1e-6:
        new_weights = new_weights * (old_std / new_std)
        
    # Also adjust mean if there's a significant shift
    mean_shift = weights.mean() - new_weights.mean()
    new_weights = new_weights + mean_shift
    
    return new_weights

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int) -> Dict[str, Any]:
    """
    Adjusts the first convolutional layer(s) of a state_dict to match a new input channel count.
    Automatically identifies layers ending in 'features_extractor.cnn.0.weight'.
    
    Args:
        state_dict: The model's state dictionary.
        new_in_channels: The target number of input channels.
        
    Returns:
        A new state dictionary with grafted weights.
    """
    # Identify input layer keys
    # SB3 CnnPolicy may have shared ('features_extractor') or separate ('pi_features_extractor', 'vf_features_extractor')
    input_keys = [k for k in state_dict.keys() if k.endswith("features_extractor.cnn.0.weight")]
    
    if not input_keys:
        logger.warning(f"No input convolutional layers found in state_dict matching '*features_extractor.cnn.0.weight'. Available keys: {list(state_dict.keys())[:10]}...")
        return state_dict

    # Check if any change is actually needed before copying
    needs_change = False
    for key in input_keys:
        if state_dict[key].shape[1] != new_in_channels:
            needs_change = True
            break
    
    if not needs_change:
        return state_dict

    new_state_dict = copy.deepcopy(state_dict)
    
    for key in input_keys:
        old_weight = state_dict[key]
        
        # Check shape: (out_channels, in_channels, kernel_h, kernel_w)
        if len(old_weight.shape) != 4:
             logger.warning(f"Skipping key '{key}': Expected 4D weight tensor, got {old_weight.shape}")
             continue
             
        old_in_channels = old_weight.shape[1]
        
        if old_in_channels == new_in_channels:
            continue
        
        logger.info(f"Grafting model weights for '{key}': {old_in_channels} -> {new_in_channels} channels")
            
        if new_in_channels < old_in_channels:
            # Shrinking: Slice the weights
            new_weight = old_weight[:, :new_in_channels, :, :]
            new_state_dict[key] = new_weight
            
        else:
            # Expanding
            out_channels, _, k_h, k_w = old_weight.shape
            
            # Initialize new tensor with small random noise
            # Using 0.01 scale to avoid destabilizing the network
            new_weight = torch.randn(out_channels, new_in_channels, k_h, k_w) * 0.01
            
            # Copy old weights
            new_weight[:, :old_in_channels, :, :] = old_weight
            
            new_state_dict[key] = new_weight
        
    return new_state_dict

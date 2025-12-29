import torch
import torch.nn.functional as F
import copy
import math
from typing import Dict, Any, Optional
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

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int, new_action_dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Adjusts the model weights to match new input channels and/or new action space dimensions.
    
    Args:
        state_dict: The model's state dictionary.
        new_in_channels: The target number of input channels for CNN.
        new_action_dim: (Optional) The target number of output actions (e.g., grid_size * grid_size).
        
    Returns:
        A new state dictionary with grafted weights.
    """
    new_state_dict = copy.deepcopy(state_dict)
    
    # --- 1. Handle Input Channels (CNN) ---
    # Identify input layer keys
    input_keys = [k for k in state_dict.keys() if k.endswith("features_extractor.cnn.0.weight")]
    
    if not input_keys:
        logger.warning(f"No input convolutional layers found in state_dict matching '*features_extractor.cnn.0.weight'.")
    else:
        for key in input_keys:
            old_weight = state_dict[key]
            
            # Check shape: (out_channels, in_channels, kernel_h, kernel_w)
            if len(old_weight.shape) != 4:
                 logger.warning(f"Skipping key '{key}': Expected 4D weight tensor, got {old_weight.shape}")
                 continue
                 
            old_in_channels = old_weight.shape[1]
            
            if old_in_channels == new_in_channels:
                continue
            
            logger.info(f"Grafting input weights for '{key}': {old_in_channels} -> {new_in_channels} channels")
                
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

    # --- 2. Handle Action/Value Heads ---
    if new_action_dim is not None:
        # PPO Action/Value heads usually contain "action_net" or "value_net"
        # We only interpolate if they look like spatial maps (square sizes)
        head_keys = [k for k in state_dict.keys() if "action_net" in k or "value_net" in k]
        
        for key in head_keys:
            old_weight = state_dict[key]
            old_dim = old_weight.shape[0]
            
            # If dimensions match, nothing to do
            if old_dim == new_action_dim:
                continue
            
            # Skip scalar heads (e.g., value_net with output dim 1)
            if old_dim == 1:
                continue

            # Check if this looks like a spatial layer
            # Is old_dim a perfect square?
            old_grid_float = math.sqrt(old_dim)
            is_square_old = old_grid_float.is_integer()
            
            # Is new_dim a perfect square?
            new_grid_float = math.sqrt(new_action_dim)
            is_square_new = new_grid_float.is_integer()
            
            if is_square_old and is_square_new:
                old_grid = int(old_grid_float)
                new_grid = int(new_grid_float)
                logger.info(f"Interpolating spatial weights for '{key}': {old_grid}x{old_grid} -> {new_grid}x{new_grid}")
                
                try:
                    # Handle Weights: (N, features)
                    if len(old_weight.shape) == 2:
                        new_state_dict[key] = interpolate_spatial_weights(old_weight, old_grid, new_grid)
                        
                    # Handle Bias: (N,) -> Reshape to (N, 1) -> Interpolate -> Flatten
                    elif len(old_weight.shape) == 1:
                        w_reshaped = old_weight.view(-1, 1)
                        w_interp = interpolate_spatial_weights(w_reshaped, old_grid, new_grid)
                        new_state_dict[key] = w_interp.view(-1)
                        
                    else:
                        logger.warning(f"Unknown shape for head layer '{key}': {old_weight.shape}. Deleting to avoid mismatch.")
                        del new_state_dict[key]
                        
                except Exception as e:
                    logger.error(f"Failed to interpolate '{key}': {e}. Deleting.")
                    del new_state_dict[key]
            else:
                # Dimensions changed but not strictly square-to-square (or scalar value net)
                # If it's a scalar value net (1, features), keep it!
                # If it's something else that mismatches, delete it.
                if old_dim == 1: 
                    # Probably value net scalar, keep it.
                    continue 
                
                logger.info(f"Dropping incompatible layer '{key}': {old_dim} -> {new_action_dim} (Not spatial squares)")
                del new_state_dict[key]
        
    return new_state_dict
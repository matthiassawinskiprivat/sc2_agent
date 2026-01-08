import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, Optional, Tuple, List
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

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int, new_action_dim: Optional[int] = None, action_handling_mode: str = "auto") -> Dict[str, Any]:
    """
    Adjusts the model weights to match new input channels and/or new action space dimensions.
    
    Args:
        state_dict: The model's state dictionary.
        new_in_channels: The target number of input channels for CNN.
        new_action_dim: (Optional) The target number of output actions.
        action_handling_mode: Strategy for action dimension change.
            - "auto": Tries interpolation first (if square), then 2x expansion.
            - "expand": Forces block repetition/randomization (e.g., 1 Head -> 4 Heads).
            - "interpolate": Forces spatial interpolation.
        
    Returns:
        A new state dictionary with grafted weights.
    """
    new_state_dict = copy.deepcopy(state_dict)
    
    # --- 1. Handle Input Channels (CNN) ---
    # ... (Same as before) ...
    # Identify input layer keys
    input_keys = [k for k in state_dict.keys() if k.endswith("features_extractor.cnn.0.weight")]
    
    if not input_keys:
        logger.warning(f"No input convolutional layers found in state_dict matching '*features_extractor.cnn.0.weight'.")
    else:
        for key in input_keys:
            old_weight = state_dict[key]
            if len(old_weight.shape) != 4: continue
            old_in_channels = old_weight.shape[1]
            if old_in_channels == new_in_channels: continue
            
            logger.info(f"Grafting input weights for '{key}': {old_in_channels} -> {new_in_channels} channels")
            if new_in_channels < old_in_channels:
                new_state_dict[key] = old_weight[:, :new_in_channels, :, :]
            else:
                out_channels, _, k_h, k_w = old_weight.shape
                new_weight = torch.randn(out_channels, new_in_channels, k_h, k_w) * 0.01
                new_weight[:, :old_in_channels, :, :] = old_weight
                new_state_dict[key] = new_weight

    # --- 2. Handle Action/Value Heads ---
    if new_action_dim is not None:
        head_keys = [k for k in state_dict.keys() if "action_net" in k or "value_net" in k]
        
        for key in head_keys:
            old_weight = state_dict[key]
            old_dim = old_weight.shape[0]
            
            if old_dim == new_action_dim: continue
            if old_dim == 1: continue # Skip scalar heads

            # Check for block expansion (multiples)
            is_expansion = (new_action_dim > old_dim) and (new_action_dim % old_dim == 0)
            
            # Check for square spatial
            old_grid_float = math.sqrt(old_dim)
            new_grid_float = math.sqrt(new_action_dim)
            is_square_scaling = old_grid_float.is_integer() and new_grid_float.is_integer()

            # Decision Logic
            use_interpolation = False
            use_expansion = False
            
            if action_handling_mode == "interpolate":
                use_interpolation = True
            elif action_handling_mode == "expand":
                use_expansion = True
            else: # auto
                if is_square_scaling and not is_expansion:
                    use_interpolation = True
                elif is_expansion:
                    use_expansion = True # Prefer expansion for clean multiples like 2x, 4x
                elif is_square_scaling:
                    use_interpolation = True
            
            # Execute
            if use_interpolation and is_square_scaling:
                logger.info(f"Interpolating spatial weights for '{key}': {old_dim} -> {new_action_dim}")
                try:
                    if len(old_weight.shape) == 2:
                        new_state_dict[key] = interpolate_spatial_weights(old_weight, int(old_grid_float), int(new_grid_float))
                    elif len(old_weight.shape) == 1:
                        w_reshaped = old_weight.view(-1, 1)
                        w_interp = interpolate_spatial_weights(w_reshaped, int(old_grid_float), int(new_grid_float))
                        new_state_dict[key] = w_interp.view(-1)
                except Exception as e:
                    logger.error(f"Interpolation failed: {e}")
                    del new_state_dict[key]
                    
            elif use_expansion and is_expansion:
                logger.info(f"Expanding action blocks for '{key}': {old_dim} -> {new_action_dim}")
                if len(old_weight.shape) == 2:
                    _, in_feat = old_weight.shape
                    new_w = torch.randn(new_action_dim, in_feat) * 0.01
                    new_w[:old_dim, :] = old_weight # Copy first block
                    new_state_dict[key] = new_w
                elif len(old_weight.shape) == 1:
                    new_b = torch.randn(new_action_dim) * 0.01
                    new_b[:old_dim] = old_weight
                    new_state_dict[key] = new_b
                else:
                    del new_state_dict[key]
            
            else:
                logger.info(f"Dropping incompatible layer '{key}'")
                del new_state_dict[key]
        
    return new_state_dict

def calculate_distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance matrix between two sets of points.
    
    Args:
        points_a: Numpy array of shape (N, 2)
        points_b: Numpy array of shape (M, 2)
        
    Returns:
        Distance matrix of shape (N, M)
    """
    # Use broadcasting to compute pairwise distances efficiently
    # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    
    # Squared Euclidean Distance: (x1-x2)^2 + (y1-y2)^2
    # Sum over the last dimension (coordinates)
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Sqrt to get Euclidean distance
    return np.sqrt(dist_sq)

def solve_optimal_assignment(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the linear sum assignment problem (optimal matching) between two sets of points.
    Minimizes the total distance.
    
    Args:
        points_a: Numpy array of shape (N, 2) - e.g., Units
        points_b: Numpy array of shape (M, 2) - e.g., Beacons
        
    Returns:
        Tuple (row_ind, col_ind) of assigned indices.
        - row_ind: Indices of assigned elements from points_a
        - col_ind: Indices of assigned elements from points_b
        
        Note: If N != M, the algorithm matches min(N, M) pairs optimally.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([]), np.array([])
        
    cost_matrix = calculate_distance_matrix(points_a, points_b)
    
    # linear_sum_assignment finds indexes that minimize total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return row_ind, col_ind
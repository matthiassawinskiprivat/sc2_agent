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

def decompose_action_dim(dim: int) -> Tuple[int, int, int]:
    """
    Heuristically decomposes an action dimension into (num_spatial_heads, grid_size, num_categorical).
    Supports common grid sizes: 128, 64, 32.
    """
    for g in [128, 64, 32]:
        g2 = g * g
        if dim >= g2:
            n = dim // g2
            c = dim % g2
            # Heuristic: if c is small (e.g. < 100), we assume this is the decomposition.
            # Most categorical head additions are just a few actions (Load, Unload, etc.)
            if c < 100:
                return n, g, c
    
    # Fallback: Treat as entirely categorical
    return 0, 0, dim

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int, new_action_dim: Optional[int] = None, 
                        action_handling_mode: str = "auto", new_grid_size: Optional[int] = None,
                        target_state_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Adjusts the model weights to match new input channels and/or new action space dimensions.
    
    Args:
        state_dict: The model's state dictionary from the checkpoint.
        new_in_channels: The target number of input channels for CNN.
        new_action_dim: (Optional) The target number of output actions.
        action_handling_mode: Strategy for action dimension change.
            - "auto": Tries interpolation first (if square), then 2x expansion.
            - "expand": Forces block repetition/randomization (e.g., 1 Head -> 4 Heads).
            - "interpolate": Forces spatial interpolation.
        new_grid_size: (Optional) The grid size of the target environment.
        target_state_dict: (Optional) The state dictionary of the current (target) model to match shapes.
        
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
    head_keys = [k for k in state_dict.keys() if ("action_net" in k or "value_net" in k) and "mlp_extractor" not in k]
    
    for key in head_keys:
        old_weight = state_dict[key]
        
        # Determine target shapes if possible
        target_shape = target_state_dict[key].shape if (target_state_dict and key in target_state_dict) else None
        
        # Handle 2D weights [output_dim, input_features]
        if len(old_weight.shape) == 2:
            old_out_dim, old_in_feat = old_weight.shape
            
            target_out_dim = target_shape[0] if target_shape else (new_action_dim if "action_net" in key else old_out_dim)
            target_in_feat = target_shape[1] if target_shape else old_in_feat
            
            if old_out_dim == target_out_dim and old_in_feat == target_in_feat:
                continue

            logger.info(f"Grafting head weights for '{key}': [{old_out_dim}, {old_in_feat}] -> [{target_out_dim}, {target_in_feat}]")
            
            # --- STEP A: Handle in_features mismatch ---
            if old_in_feat != target_in_feat:
                logger.info(f"Resizing in_features for '{key}': {old_in_feat} -> {target_in_feat}")
                if target_in_feat < old_in_feat:
                    # Truncate features
                    temp_weight = old_weight[:, :target_in_feat]
                else:
                    # Pad features with small random values
                    temp_weight = torch.randn(old_out_dim, target_in_feat) * 0.01
                    temp_weight[:, :old_in_feat] = old_weight
                
                # Update current working weight for next steps
                current_weight = temp_weight
                curr_out_dim = old_out_dim
            else:
                current_weight = old_weight
                curr_out_dim = old_out_dim

            # --- STEP B: Handle output_dim (Actions) mismatch ---
            if curr_out_dim != target_out_dim:
                # Reuse existing spatial handling logic but on current_weight
                
                # Check for Pure Square Interpolation
                n_old, g_old, c_old = decompose_action_dim(curr_out_dim)
                n_new, g_new, c_new = decompose_action_dim(target_out_dim)
                if new_grid_size:
                    g_new = new_grid_size
                    n_new = target_out_dim // (g_new * g_new)
                    c_new = target_out_dim % (g_new * g_new)

                # Decision logic simplified for hybrid cases
                use_interpolation = (g_old > 0 and g_new > 0 and g_new != g_old) or (action_handling_mode == "interpolate")
                
                if use_interpolation and g_old > 0:
                    logger.info(f"Interpolating spatial heads for '{key}': {n_old}x{g_old}^2 -> {n_new}x{g_new}^2")
                    try:
                        # 1. Prepare new weight tensor
                        new_w = torch.randn(target_out_dim, target_in_feat) * 0.01
                        
                        # 2. Interpolate each spatial head
                        num_heads_to_transfer = min(n_old, n_new)
                        for i in range(num_heads_to_transfer):
                            o_s, o_e = i * (g_old * g_old), (i + 1) * (g_old * g_old)
                            n_s, n_e = i * (g_new * g_new), (i + 1) * (g_new * g_new)
                            
                            head = current_weight[o_s:o_e]
                            interp_head = interpolate_spatial_weights(head, g_old, g_new)
                            new_w[n_s:n_e, :] = interp_head
                                
                        # 3. Handle Categorical part
                        o_cat_s = n_old * (g_old * g_old)
                        n_cat_s = n_new * (g_new * g_new)
                        num_cat_to_transfer = min(c_old, c_new)
                        if num_cat_to_transfer > 0:
                            new_w[n_cat_s : n_cat_s + num_cat_to_transfer, :] = current_weight[o_cat_s : o_cat_s + num_cat_to_transfer, :]
                        
                        new_state_dict[key] = new_w
                    except Exception as e:
                        logger.error(f"Hybrid grafting failed for '{key}': {e}")
                        del new_state_dict[key]
                else:
                    # Simple resize (truncate or append)
                    if target_out_dim < curr_out_dim:
                        new_state_dict[key] = current_weight[:target_out_dim, :]
                    else:
                        new_w = torch.randn(target_out_dim, target_in_feat) * 0.01
                        new_w[:curr_out_dim, :] = current_weight
                        new_state_dict[key] = new_w
            else:
                new_state_dict[key] = current_weight

        # Handle 1D biases [output_dim]
        elif len(old_weight.shape) == 1:
            old_dim = old_weight.shape[0]
            target_dim = target_shape[0] if target_shape else (new_action_dim if "action_net" in key else old_dim)
            
            if old_dim != target_dim:
                logger.info(f"Grafting head biases for '{key}': {old_dim} -> {target_dim}")
                if target_dim < old_dim:
                    new_state_dict[key] = old_weight[:target_dim]
                else:
                    new_b = torch.ones(target_dim) * (5.0 if "action_net" in key else 0.0)
                    new_b[:old_dim] = old_weight
                    new_state_dict[key] = new_b
        
    return new_state_dict
        
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
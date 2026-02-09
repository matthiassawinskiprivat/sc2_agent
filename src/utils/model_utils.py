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
                        action_handling_mode: str = "auto", new_grid_size: Optional[int] = None) -> Dict[str, Any]:
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
        new_grid_size: (Optional) The grid size of the target environment.
        
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
    if new_action_dim is not None:
        head_keys = [k for k in state_dict.keys() if ("action_net" in k or "value_net" in k) and "mlp_extractor" not in k]
        
        for key in head_keys:
            old_weight = state_dict[key]
            old_dim = old_weight.shape[0]
            
            if old_dim == new_action_dim: continue
            if old_dim == 1: continue # Skip scalar heads

            # --- DECOMPOSITION ---
            n_old, g_old, c_old = decompose_action_dim(old_dim)
            n_new, g_new, c_new = decompose_action_dim(new_action_dim)
            
            # Override with explicit grid size if provided
            if new_grid_size:
                g_new = new_grid_size
                n_new = new_action_dim // (g_new * g_new)
                c_new = new_action_dim % (g_new * g_new)

            logger.debug(f"Decomposed '{key}': Old({n_old}x{g_old}^2 + {c_old}) -> New({n_new}x{g_new}^2 + {c_new})")

            # Check for block expansion (multiples)
            is_expansion = (new_action_dim > old_dim) and (new_action_dim % old_dim == 0)
            
            # Check for pure square spatial (legacy logic check)
            old_grid_float = math.sqrt(old_dim)
            new_grid_float = math.sqrt(new_action_dim)
            is_square_scaling = old_grid_float.is_integer() and new_grid_float.is_integer()

            # Decision Logic
            use_interpolation = False
            use_expansion = False
            use_hybrid = False # Interpolate spatial, then handle categorical
            
            if action_handling_mode == "interpolate":
                use_interpolation = True
            elif action_handling_mode == "expand":
                use_expansion = True
            elif g_old > 0 and g_new > 0 and g_new != g_old:
                # Grid size changed! We MUST interpolate the spatial components.
                use_hybrid = True
            else: # auto
                if is_square_scaling and not is_expansion:
                    use_interpolation = True
                elif is_expansion:
                    use_expansion = True # Prefer expansion for clean multiples like 2x, 4x
                elif is_square_scaling:
                    use_interpolation = True
            
            # --- EXECUTION ---
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

            elif use_hybrid:
                logger.info(f"Hybrid Grafting for '{key}': Interpolating {n_old} spatial heads ({g_old}->{g_new}) and appending/truncating categorical.")
                try:
                    in_feat = old_weight.shape[1] if len(old_weight.shape) == 2 else 1
                    
                    # 1. Prepare new weight tensor
                    if len(old_weight.shape) == 2:
                        new_w = torch.randn(new_action_dim, in_feat) * 0.01
                    else:
                        new_w = torch.randn(new_action_dim) * 0.01
                    
                    # 2. Interpolate each spatial head
                    num_heads_to_transfer = min(n_old, n_new)
                    if n_old > 0 and n_new > 0:
                        for i in range(num_heads_to_transfer):
                            old_start = i * (g_old * g_old)
                            old_end = old_start + (g_old * g_old)
                            new_start = i * (g_new * g_new)
                            new_end = new_start + (g_new * g_new)
                            
                            head_weights = old_weight[old_start:old_end]
                            if len(old_weight.shape) == 1:
                                head_weights = head_weights.view(-1, 1)
                                
                            interp_head = interpolate_spatial_weights(head_weights, g_old, g_new)
                            
                            if len(old_weight.shape) == 2:
                                new_w[new_start:new_end, :] = interp_head
                            else:
                                new_w[new_start:new_end] = interp_head.view(-1)
                                
                    # 3. Handle Categorical part
                    # We copy old categorical actions to the end of the new spatial block
                    old_cat_start = n_old * (g_old * g_old)
                    new_cat_start = n_new * (g_new * g_new)
                    
                    num_cat_to_transfer = min(c_old, c_new)
                    if num_cat_to_transfer > 0:
                        if len(old_weight.shape) == 2:
                            new_w[new_cat_start:new_cat_start+num_cat_to_transfer, :] = old_weight[old_cat_start:old_cat_start+num_cat_to_transfer, :]
                        else:
                            new_w[new_cat_start:new_cat_start+num_cat_to_transfer] = old_weight[old_cat_start:old_cat_start+num_cat_to_transfer]
                    
                    # 4. Loud Bias for COMPLETELY new actions (categorical or new heads)
                    if len(old_weight.shape) == 1:
                        # Find indices that were NOT copied
                        # We just apply it to everything from new_cat_start + num_cat_to_transfer onwards
                        new_w[new_cat_start+num_cat_to_transfer:] = 5.0
                        
                    new_state_dict[key] = new_w
                    
                except Exception as e:
                    logger.error(f"Hybrid grafting failed for '{key}': {e}")
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

            elif new_action_dim > old_dim:
                # CATEGORICAL APPEND: Handle adding categorical heads to existing spatial heads
                logger.info(f"Appending categorical actions for '{key}': {old_dim} -> {new_action_dim} (Loud Bias: +5.0)")
                if len(old_weight.shape) == 2:
                    out_dim, in_feat = old_weight.shape
                    # For weights, we use small random values for new indices
                    new_w = torch.randn(new_action_dim, in_feat) * 0.01
                    new_w[:old_dim, :] = old_weight
                    new_state_dict[key] = new_w
                elif len(old_weight.shape) == 1:
                    # For biases, we use a LOUD (+5.0) value for the new categorical slots
                    new_b = torch.ones(new_action_dim) * 5.0
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
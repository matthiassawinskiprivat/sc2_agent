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
    """
    if weights.shape[0] != old_size * old_size:
        raise ValueError(f"Weights first dimension {weights.shape[0]} does not match spatial size {old_size}x{old_size}")

    # weights: (spatial, features) -> (features, spatial)
    # Then reshape to (1, features, old_size, old_size) for 2D interpolation
    in_features = weights.shape[1]
    x = weights.t().view(1, in_features, old_size, old_size)
    
    # Perform bilinear interpolation
    x_new = F.interpolate(x, size=(new_size, new_size), mode='bilinear', align_corners=True)
    
    # Reshape back to (new_size*new_size, in_features)
    new_weights = x_new.view(in_features, new_size * new_size).t()
    
    # Preservation of statistics
    old_std = weights.std()
    new_std = new_weights.std()
    if new_std > 1e-6:
        new_weights = new_weights * (old_std / new_std)
        
    mean_shift = weights.mean() - new_weights.mean()
    new_weights = new_weights + mean_shift
    
    return new_weights

def decompose_action_dim(dim: int) -> Tuple[int, int, int]:
    """
    Heuristically decomposes an action dimension into (num_spatial_heads, grid_size, num_categorical).
    Supports common grid sizes: 128, 64, 32, 16.
    """
    for g in [128, 64, 32, 16]:
        g2 = g * g
        if dim >= g2:
            n = dim // g2
            c = dim % g2
            if c < 100:
                return n, g, c
    return 0, 0, dim

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int, new_action_dim: Optional[int] = None, 
                        action_handling_mode: str = "auto", new_grid_size: Optional[int] = None,
                        target_state_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Surgically adjusts model weights to match new architectures.
    """
    new_state_dict = copy.deepcopy(state_dict)
    
    # --- 1. Handle Input Channels (CNN & Tactical CNN) ---
    # Matches first conv weights on global/tactical stacks
    _suffixes = (
        "cnn.0.weight",
        "cnn_trunk.0.weight",
        "tactical_cnn_to16.0.weight",
        "tactical_cnn_legacy.0.weight",
    )
    input_keys = [k for k in state_dict.keys() if any(k.endswith(s) for s in _suffixes)]
    
    for key in input_keys:
        old_weight = state_dict[key]
        if len(old_weight.shape) != 4: continue
        
        # Determine target channels from target_state_dict if available, else use argument
        target_in_channels = target_state_dict[key].shape[1] if (target_state_dict and key in target_state_dict) else new_in_channels
        old_in_channels = old_weight.shape[1]
        
        if old_in_channels == target_in_channels: continue
        
        logger.info(f"Grafting input weights for '{key}': {old_in_channels} -> {target_in_channels} channels")
        if target_in_channels < old_in_channels:
            new_state_dict[key] = old_weight[:, :target_in_channels, :, :]
        else:
            out_channels, _, k_h, k_w = old_weight.shape
            new_weight = torch.randn(out_channels, target_in_channels, k_h, k_w) * 0.01
            new_weight[:, :old_in_channels, :, :] = old_weight
            new_state_dict[key] = new_weight

    # --- 2. Handle Fusion Layer (The "Brain") ---
    # Matches 'linear.0.weight'
    fusion_keys = [k for k in state_dict.keys() if k.endswith("linear.0.weight")]
    for key in fusion_keys:
        if target_state_dict and key in target_state_dict:
            old_w = state_dict[key]
            new_w = target_state_dict[key]
            
            if old_w.shape == new_w.shape: continue
            
            logger.info(f"Surgically grafting Fusion Layer '{key}': {old_w.shape} -> {new_w.shape}")
            
            # Map Structure: [Global_Feat, (Tactical_Feat), Command_Feat]
            # Command is always the LAST 10 dimensions.
            old_in_dim = old_w.shape[1]
            new_in_dim = new_w.shape[1]
            
            cmd_dim = 10
            
            # Create fresh weights based on target (randomized)
            grafted_w = torch.randn_like(new_w) * 0.01
            
            # 1. Transfer Global (First chunk)
            # If we added tactical, global is still at the beginning.
            # We assume Global size doesn't change unless grid size changed (handled by linear resize if needed)
            global_size = min(old_in_dim - cmd_dim, new_in_dim - cmd_dim)
            grafted_w[:, :global_size] = old_w[:, :global_size]
            
            # 2. Transfer Command (Last chunk)
            grafted_w[:, -cmd_dim:] = old_w[:, -cmd_dim:]
            
            new_state_dict[key] = grafted_w
            
            # Handle bias
            bias_key = key.replace(".weight", ".bias")
            if bias_key in state_dict and bias_key in target_state_dict:
                new_state_dict[bias_key] = state_dict[bias_key] # features_dim is constant

    # --- 3. Handle Action/Value Heads ---
    head_keys = [
        k for k in state_dict.keys()
        if ("action_net" in k or "value_net" in k or "command_head" in k or "spatial_decoder" in k)
        and "mlp_extractor" not in k
    ]
    
    for key in head_keys:
        old_weight = state_dict[key]
        target_shape = target_state_dict[key].shape if (target_state_dict and key in target_state_dict) else None
        
        if len(old_weight.shape) == 2:
            old_out_dim, old_in_feat = old_weight.shape
            target_out_dim = target_shape[0] if target_shape else (new_action_dim if "action_net" in key else old_out_dim)
            target_in_feat = target_shape[1] if target_shape else old_in_feat
            
            if old_out_dim == target_out_dim and old_in_feat == target_in_feat:
                continue

            logger.info(f"Grafting head weights for '{key}': [{old_out_dim}, {old_in_feat}] -> [{target_out_dim}, {target_in_feat}]")
            
            # A. Handle in_features mismatch (Linear resize)
            if old_in_feat != target_in_feat:
                if target_in_feat < old_in_feat:
                    current_weight = old_weight[:, :target_in_feat]
                else:
                    current_weight = torch.randn(old_out_dim, target_in_feat) * 0.01
                    current_weight[:, :old_in_feat] = old_weight
                curr_out_dim = old_out_dim
            else:
                current_weight = old_weight
                curr_out_dim = old_out_dim

            # B. Handle output_dim (Actions) mismatch (Spatial Interpolation)
            if curr_out_dim != target_out_dim:
                n_old, g_old, c_old = decompose_action_dim(curr_out_dim)
                n_new, g_new, c_new = decompose_action_dim(target_out_dim)
                if new_grid_size:
                    g_new = new_grid_size
                    n_new = target_out_dim // (g_new * g_new)
                    c_new = target_out_dim % (g_new * g_new)

                use_interpolation = (g_old > 0 and g_new > 0 and g_new != g_old) or (action_handling_mode == "interpolate")
                
                if use_interpolation and g_old > 0:
                    logger.info(f"Interpolating spatial heads: {n_old}x{g_old}^2 -> {n_new}x{g_new}^2")
                    new_w = torch.randn(target_out_dim, target_in_feat) * 0.01
                    num_heads_to_transfer = min(n_old, n_new)
                    for i in range(num_heads_to_transfer):
                        o_s, o_e = i * (g_old * g_old), (i + 1) * (g_old * g_old)
                        n_s, n_e = i * (g_new * g_new), (i + 1) * (g_new * g_new)
                        new_w[n_s:n_e, :] = interpolate_spatial_weights(current_weight[o_s:o_e], g_old, g_new)
                    
                    # Categorical Transfer
                    o_cat_s = n_old * (g_old * g_old)
                    n_cat_s = n_new * (g_new * g_new)
                    num_cat = min(c_old, c_new)
                    if num_cat > 0:
                        new_w[n_cat_s:n_cat_s+num_cat, :] = current_weight[o_cat_s:o_cat_s+num_cat, :]
                    new_state_dict[key] = new_w
                else:
                    if target_out_dim < curr_out_dim:
                        new_state_dict[key] = current_weight[:target_out_dim, :]
                    else:
                        new_w = torch.randn(target_out_dim, target_in_feat) * 0.01
                        new_w[:curr_out_dim, :] = current_weight
                        new_state_dict[key] = new_w
            else:
                new_state_dict[key] = current_weight

        elif len(old_weight.shape) == 1:
            old_dim = old_weight.shape[0]
            target_dim = target_shape[0] if target_shape else old_dim
            if old_dim != target_dim:
                if target_dim < old_dim:
                    new_state_dict[key] = old_weight[:target_dim]
                else:
                    new_b = torch.ones(target_dim) * (5.0 if "action_net" in key else 0.0)
                    new_b[:old_dim] = old_weight
                    new_state_dict[key] = new_b
        
    # --- 4. Handle Missing Keys (e.g., brand new tactical_cnn) ---
    if target_state_dict:
        for key in target_state_dict.keys():
            if key not in new_state_dict:
                logger.info(f"Initializing brand new component '{key}'")
                new_state_dict[key] = target_state_dict[key]

    return new_state_dict

def calculate_distance_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def solve_optimal_assignment(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([]), np.array([])
    cost_matrix = calculate_distance_matrix(points_a, points_b)
    return linear_sum_assignment(cost_matrix)

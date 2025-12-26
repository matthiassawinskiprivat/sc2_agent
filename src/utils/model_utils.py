import torch
import copy
from typing import Dict, Any
from loguru import logger

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

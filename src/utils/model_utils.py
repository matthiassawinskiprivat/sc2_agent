import torch
import copy
from typing import Dict, Any
from loguru import logger

def graft_model_weights(state_dict: Dict[str, Any], new_in_channels: int, 
                        input_layer_name: str = "features_extractor.cnn.0.weight") -> Dict[str, Any]:
    """
    Adjusts the first convolutional layer of a state_dict to match a new input channel count.
    
    Args:
        state_dict: The model's state dictionary.
        new_in_channels: The target number of input channels.
        input_layer_name: Key of the input layer weight in the state dict.
        
    Returns:
        A new state dictionary with grafted weights.
    """
    if input_layer_name not in state_dict:
        # Fallback search for common names or throw error?
        # SB3 NatureCNN might be different? 
        # Usually it is 'features_extractor.cnn.0.weight' for CnnPolicy with NatureCNN
        raise ValueError(f"Input layer '{input_layer_name}' not found in state_dict. Available keys: {list(state_dict.keys())[:5]}...")
    
    old_weight = state_dict[input_layer_name]
    
    # Check shape: (out_channels, in_channels, kernel_h, kernel_w)
    if len(old_weight.shape) != 4:
         raise ValueError(f"Expected 4D weight tensor for '{input_layer_name}', got {old_weight.shape}")
         
    old_in_channels = old_weight.shape[1]
    
    if old_in_channels == new_in_channels:
        return state_dict
    
    logger.info(f"Grafting model weights: {old_in_channels} -> {new_in_channels} channels for layer '{input_layer_name}'")
        
    # Create copy to avoid modifying original
    new_state_dict = copy.deepcopy(state_dict)
    
    if new_in_channels < old_in_channels:
        # Shrinking: Slice the weights
        new_weight = old_weight[:, :new_in_channels, :, :]
        new_state_dict[input_layer_name] = new_weight
        
    else:
        # Expanding
        out_channels, _, k_h, k_w = old_weight.shape
        
        # Initialize new tensor with small random noise
        # Using 0.01 scale to avoid destabilizing the network
        new_weight = torch.randn(out_channels, new_in_channels, k_h, k_w) * 0.01
        
        # Copy old weights
        new_weight[:, :old_in_channels, :, :] = old_weight
        
        new_state_dict[input_layer_name] = new_weight
        
    return new_state_dict

"""
Shared Observation Transformer for Multiple Agents
Transforms SC2 observations once and shares among all agents
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .observation_transformer import ObservationTransformer


class SharedObservationTransformer:
    """Transforms observations once and shares among multiple agents"""
    
    def __init__(self, map_size: tuple = (128, 128), unit_data_file: str = None):
        """
        Initialize shared transformer
        
        Args:
            map_size: (width, height) for output maps
            unit_data_file: Path to unit data JSON file
        """
        self.transformer = ObservationTransformer(
            map_size=map_size,
            unit_data_file=unit_data_file
        )
        
        # Cache for transformed observations
        self.current_observation = None
        self.transformed_observation = None
        self.last_observation_id = None
        
    def transform_observation(self, observation: Dict[str, Any], 
                            observation_id: Any = None, player_id: int = 1) -> np.ndarray:
        """
        Transform observation if it's new, otherwise return cached result
        
        Args:
            observation: Raw SC2 observation
            observation_id: Unique identifier for the observation (for caching)
            player_id: Player ID for allied/enemy classification
            
        Returns:
            Transformed observation [6×H×W] for all 6 core layers
        """
        # Check if this is a new observation
        if observation_id is not None and observation_id == self.last_observation_id:
            return self.transformed_observation
        
        # Transform the observation
        self.current_observation = observation
        self.transformed_observation = self.transformer.transform(observation)
        self.last_observation_id = observation_id
        
        return self.transformed_observation
    
    def get_full_layer_stack(self) -> Optional[np.ndarray]:
        """Get the full stack of all 6 layers"""
        return self.transformed_observation
    
    def get_layer_by_index(self, index: int) -> Optional[np.ndarray]:
        """Get a specific layer by index (0-5)"""
        if self.transformed_observation is not None and 0 <= index < 6:
            return self.transformed_observation[index]
        return None
    
    def get_multiple_layers(self, indices: List[int]) -> Optional[np.ndarray]:
        """Get multiple layers by indices"""
        if self.transformed_observation is None:
            return None
        
        valid_indices = [idx for idx in indices if 0 <= idx < 6]
        if not valid_indices:
            return None
        
        return self.transformed_observation[valid_indices]
    
    def clear_cache(self):
        """Clear the observation cache"""
        self.current_observation = None
        self.transformed_observation = None
        self.last_observation_id = None
    
    def get_transformer_config(self) -> Dict[str, Any]:
        """Get the underlying transformer's configuration"""
        return self.transformer.get_config()
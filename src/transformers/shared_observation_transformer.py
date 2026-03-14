"""
Shared Observation Transformer for Multiple Agents
Transforms SC2 observations once and shares among all agents
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from .observation_transformer import ObservationTransformer


class SharedObservationTransformer:
    """Transforms observations once and shares among multiple agents"""

    def __init__(self, map_size: tuple = (128, 128), unit_data_file: str = None, training_conf: Dict = None):
        """
        Initialize shared transformer
        
        Args:
            map_size: (width, height) for output maps
            unit_data_file: Path to unit data JSON file
            training_conf: Dictionary with training-specific settings
        """
        self.transformer = ObservationTransformer(
            map_size=map_size,
            unit_data_file=unit_data_file,
            training_conf=training_conf
        )

        # Cache for transformed observations
        self.current_observation = None
        self.transformed_observation = None
        self.last_observation_id = None

    def transform_observation(self, observation: Dict[str, Any],
                              observation_id: Any = None, player_id: int = 1,
                              active_unit: Any = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transform observation if it's new, otherwise return cached result
        """
        # Check if this is a new observation
        if observation_id is not None and observation_id == self.last_observation_id:
            # Multi-agent cache: if we have a cached dict, we check if the active_unit is the same.
            # However, for UI and simple cases, just returning the cache is better.
            if self.transformed_observation is not None:
                return self.transformed_observation

        # Transform the observation
        self.current_observation = observation
        global_result = self.transformer.transform(observation)
        
        if isinstance(global_result, dict):
            # Universal Architecture: Add tactical stream
            tactical_obs = self.transformer.transform_tactical(observation, active_unit)
            self.transformed_observation = {
                "map": global_result["map"],
                "tactical": tactical_obs,
                "command": global_result["command"]
            }
        else:
            self.transformed_observation = global_result
            
        self.last_observation_id = observation_id
        return self.transformed_observation

    def get_full_layer_stack(self) -> Optional[np.ndarray]:
        """Get the full stack of all configured layers"""
        return self.transformed_observation

    def get_layer_by_index(self, index: int) -> Optional[np.ndarray]:
        """Get a specific layer by index"""
        if self.transformed_observation is not None:
            num_layers = self.transformed_observation.shape[0]
            if 0 <= index < num_layers:
                return self.transformed_observation[index]
        return None

    def get_multiple_layers(self, indices: List[int]) -> Optional[np.ndarray]:
        """Get multiple layers by indices"""
        if self.transformed_observation is None:
            return None

        num_layers = self.transformed_observation.shape[0]
        valid_indices = [idx for idx in indices if 0 <= idx < num_layers]
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

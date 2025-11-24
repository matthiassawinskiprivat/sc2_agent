"""
StarCraft 2 Unit Control Agent
Clean architecture with proper separation of concerns
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional

from ..transformers.observation_transformer import ObservationTransformer
from ..utils.layer_constants import LayerConstants


class UnitControlAgent:
    """Pure unit control agent without UI responsibilities"""
    
    def __init__(self, cnn_input_size: tuple[int, int] = (128, 128)):
        """
        Initialize the unit control agent
        
        Args:
            cnn_input_size: (width, height) for CNN input
        """
        self.cnn_width, self.cnn_height = cnn_input_size
        
        # Agent state - receives transformed observations from shared transformer
        self.current_observation = None
        self.transformed_observation = None
        self.frame_count = 0
        self.last_process_time = time.time()
        
        # Performance tracking
        self.processing_times = []
        self.avg_processing_time = 0.0
        self.current_fps = 0.0

    def process_transformed_observation(self, transformed_observation: np.ndarray, 
                                      observation_metadata: Dict[str, Any] = None) -> np.ndarray:
        """
        Process a pre-transformed observation
        
        Args:
            transformed_observation: Already transformed observation [6×H×W]
            observation_metadata: Optional metadata about the observation
            
        Returns:
            The transformed observation data
        """
        start_time = time.time()
        
        try:
            # Update state
            self.transformed_observation = transformed_observation
            self.frame_count += 1
            
            # Update performance metrics
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            if len(self.processing_times) > 100:  # Keep last 100 times
                self.processing_times.pop(0)
            self.avg_processing_time = np.mean(self.processing_times)
            
            return transformed_observation
            
        except Exception as e:
            print(f"Error processing transformed observation: {e}")
            return None
    
    def get_observation_statistics(self) -> Dict[str, Any]:
        """Get statistics about current observation"""
        if self.transformed_observation is None:
            return {}
        
        stats = {
            'shape': self.transformed_observation.shape,
            'dtype': str(self.transformed_observation.dtype),
            'processing_time_ms': self.avg_processing_time * 1000,
            'frame_count': self.frame_count,
            'min_values': {},
            'max_values': {},
            'mean_values': {},
            'non_zero_pixels': {}
        }
        
        # Get layer statistics
        for layer_name, layer_idx in LayerConstants.LAYERS.items():
            if layer_idx < self.transformed_observation.shape[0]:
                layer_data = self.transformed_observation[layer_idx]
                stats['min_values'][layer_name] = float(np.min(layer_data))
                stats['max_values'][layer_name] = float(np.max(layer_data))
                stats['mean_values'][layer_name] = float(np.mean(layer_data))
                stats['non_zero_pixels'][layer_name] = int(np.count_nonzero(layer_data))
        
        return stats
    
    def select_layers(self, layer_indices: List[int]) -> np.ndarray:
        """
        Select specific layers from the transformed observation by index
        
        Args:
            layer_indices: List of layer indices (0-5) to select
            
        Returns:
            Array with selected layers only
        """
        if self.transformed_observation is None:
            return None
        
        # Validate indices (must be 0-5 for the 6 core layers)
        valid_indices = []
        for idx in layer_indices:
            if 0 <= idx < 6:
                valid_indices.append(idx)
        
        if not valid_indices:
            return None
        
        # Select layers
        return self.transformed_observation[valid_indices]
    
    def get_full_layer_stack(self) -> np.ndarray:
        """Get the full stack of all available layers"""
        return self.transformed_observation
    
    def reset(self):
        """Reset agent state"""
        self.current_observation = None
        self.transformed_observation = None
        self.frame_count = 0
        self.processing_times = []
        self.avg_processing_time = 0.0
    
    #def get_config(self) -> Dict[str, Any]:
    #    """Get current configuration"""
    #    return {
    #        'cnn_input_size': (self.cnn_width, self.cnn_height),
    #        'unit_data_file': self.unit_data_file,
    #        'transformer_config': self.obs_transformer.get_config()
    #    }


class UnitControlAgentWithUI:
    """Extended agent with UI capabilities (separate from core agent)"""
    
    def __init__(self, cnn_input_size: tuple[int, int] = (128, 128),
                 display_size: tuple[int, int] = (800, 600),
                 unit_data_file: str = None):
        """
        Initialize agent with UI support
        
        Args:
            cnn_input_size: (width, height) for CNN input
            display_size: (width, height) for UI display
            unit_data_file: Path to unit data JSON file
        """
        self.core_agent = UnitControlAgent(cnn_input_size, unit_data_file)
        self.display_size = display_size
        
        # UI components (optional)
        self.ui_transformer = None
        self.ui_enabled = False
        
    def enable_ui(self):
        """Enable UI functionality"""
        try:
            from ..transformers.ui_transformer import RealTimeUITransformer
            self.ui_transformer = RealTimeUITransformer(display_size=self.display_size)
            self.ui_enabled = True
            print("UI enabled successfully")
        except ImportError:
            print("UI dependencies not available - UI disabled")
            self.ui_enabled = False
    
    def process_observation(self, observation: Dict[str, Any], player_id: int = 1) -> np.ndarray:
        """Process observation with optional UI updates"""
        result = self.core_agent.process_observation(observation, player_id)
        
        # Update UI if enabled
        if self.ui_enabled and self.ui_transformer and result is not None:
            # This would be called by external UI manager in proper architecture
            pass
        print(f"Transformed observation shape: {result.shape}")
        return result
    
    def get_ui_frame(self) -> Optional[np.ndarray]:
        """Get UI frame for display"""
        if not self.ui_enabled or not self.ui_transformer:
            return None
        
        if self.core_agent.transformed_observation is not None:
            return self.ui_transformer.update_display(
                self.core_agent.transformed_observation,
                fps=self.get_fps()
            )
        return None
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if self.core_agent.avg_processing_time > 0:
            return 1.0 / self.core_agent.avg_processing_time
        return 0.0
    
    # Delegate other methods to core agent
    def __getattr__(self, name):
        return getattr(self.core_agent, name)


class HierarchicalUnitControlAgent:
    """Future hierarchical agent that will manage multiple unit control agents"""
    
    def __init__(self, cnn_input_size: tuple[int, int] = (128, 128)):
        self.cnn_input_size = cnn_input_size
        self.unit_agents = {}  # Will contain multiple UnitControlAgent instances
        self.master_agent = None  # Will be implemented later
        
    def create_unit_agent(self, agent_id: str) -> UnitControlAgent:
        """Create a new unit control agent"""
        agent = UnitControlAgent(self.cnn_input_size)
        self.unit_agents[agent_id] = agent
        return agent
    
    def assign_units_to_agent(self, unit_ids: List[int], agent_id: str):
        """Assign specific units to a unit control agent"""
        # Future implementation
        pass
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation through hierarchical structure"""
        results = {}
        
        # Process with each unit agent
        for agent_id, agent in self.unit_agents.items():
            result = agent.process_observation(observation)
            if result is not None:
                results[agent_id] = result
        
        return results
    
    def get_master_decisions(self) -> Dict[str, Any]:
        """Get strategic decisions from master agent"""
        # Future implementation
        return {}
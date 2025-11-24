"""
Hierarchical Agent System
Manages shared observation transformation and multiple unit control agents
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple

from .unit_control_agent import UnitControlAgent
from ..transformers.shared_observation_transformer import SharedObservationTransformer
from ..utils.layer_constants import LayerConstants


class HierarchicalAgentSystem:
    """Manages shared observation transformation and multiple agents"""
    
    def __init__(self, map_size: Tuple[int, int] = (128, 128), 
                 unit_data_file: str = None):
        """
        Initialize hierarchical agent system
        
        Args:
            map_size: (width, height) for output maps
            unit_data_file: Path to unit data JSON file
        """
        # Shared observation transformer
        self.shared_transformer = SharedObservationTransformer(
            map_size=map_size,
            unit_data_file=unit_data_file
        )
        
        # Multiple unit control agents
        self.unit_agents = {}
        self.agent_configs = {}
        
        # System state
        self.current_observation_id = 0
        self.frame_count = 0
        self.system_stats = {
            'total_transformations': 0,
            'total_agent_processes': 0,
            'avg_transformation_time': 0.0
        }
    
    def create_unit_agent(self, agent_id: str, cnn_input_size: Tuple[int, int] = (128, 128),
                         layer_indices: List[int] = None) -> UnitControlAgent:
        """
        Create a new unit control agent
        """
        if agent_id in self.unit_agents:
            raise ValueError(f"Agent with ID '{agent_id}' already exists")

        # Create agent
        agent = UnitControlAgent(cnn_input_size=cnn_input_size)

        # FIX: Use correct number of layers from constants
        total_layers = LayerConstants.get_num_layers()

        # Store agent and config
        self.unit_agents[agent_id] = agent
        self.agent_configs[agent_id] = {
            # Default to ALL layers if none specified
            'layer_indices': layer_indices or list(range(total_layers)),
            'cnn_input_size': cnn_input_size
        }

        print(f"Created unit agent '{agent_id}' with {len(self.agent_configs[agent_id]['layer_indices'])} layers")
        return agent

    def process_observation(self, observation: Dict[str, Any], player_id: int = 1) -> Dict[str, np.ndarray]:
        """
        Process observation through the hierarchical system
        """
        start_time = time.time()

        # Transform observation once using shared transformer
        obs_id = f"obs_{self.current_observation_id}_{int(time.time() * 1000)}"
        full_transformed = self.shared_transformer.transform_observation(
            observation, observation_id=obs_id, player_id=player_id
        )

        if full_transformed is None:
            return {}

        transformation_time = time.time() - start_time

        # Update system stats
        self.system_stats['total_transformations'] += 1
        self.system_stats['avg_transformation_time'] = (
            (self.system_stats['avg_transformation_time'] * (self.system_stats['total_transformations'] - 1) + transformation_time) /
            self.system_stats['total_transformations']
        )

        # Process with each unit agent
        results = {}

        for agent_id, agent in self.unit_agents.items():
            # Get the layers this agent needs
            layer_indices = self.agent_configs[agent_id]['layer_indices']

            # Extract specific layers for this agent
            total_layers = full_transformed.shape[0]

            # Safety check if transformer outputs fewer layers than expected
            if len(layer_indices) > total_layers:
                 print(f"Warning: Agent expects {len(layer_indices)} layers but transformer produced {total_layers}.")
                 # Fallback: slice what we can
                 agent_data = full_transformed
            elif len(layer_indices) == total_layers:
                # Agent uses all layers
                agent_data = full_transformed
            else:
                # Agent uses subset of layers
                agent_data = full_transformed[layer_indices]

            # Process with agent
            processed_data = agent.process_transformed_observation(agent_data)

            if processed_data is not None:
                results[agent_id] = processed_data
                self.system_stats['total_agent_processes'] += 1

        self.current_observation_id += 1
        self.frame_count += 1

        return results

    # ... (Rest of the methods remain unchanged) ...
    def get_agent_data(self, agent_id: str) -> Optional[np.ndarray]:
        agent = self.unit_agents.get(agent_id)
        if agent: return agent.transformed_observation
        return None

    def get_full_layer_stack(self) -> Optional[np.ndarray]:
        return self.shared_transformer.get_full_layer_stack()

    def get_system_statistics(self) -> Dict[str, Any]:
        stats = {
            'frame_count': self.frame_count,
            'num_agents': len(self.unit_agents),
            'system_stats': self.system_stats.copy(),
            'transformer_config': self.shared_transformer.get_transformer_config(),
            'agent_configs': self.agent_configs.copy()
        }
        agent_stats = {}
        for agent_id, agent in self.unit_agents.items():
            agent_stats[agent_id] = agent.get_observation_statistics()
        stats['agent_statistics'] = agent_stats
        return stats

    def reset_system(self):
        self.shared_transformer.clear_cache()
        for agent in self.unit_agents.values():
            agent.reset()
        self.current_observation_id = 0
        self.frame_count = 0
        self.system_stats = {'total_transformations': 0, 'total_agent_processes': 0, 'avg_transformation_time': 0.0}

    def remove_agent(self, agent_id: str):
        if agent_id in self.unit_agents:
            del self.unit_agents[agent_id]
            del self.agent_configs[agent_id]

    def list_agents(self) -> List[str]:
        return list(self.unit_agents.keys())
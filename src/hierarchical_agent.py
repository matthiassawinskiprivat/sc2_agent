"""
Hierarchical Agent System
Manages shared observation transformation and multiple unit control agents
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from src.agent.unit_control_agent import UnitControlAgent
from src.transformers.shared_observation_transformer import SharedObservationTransformer


class HierarchicalAgentSystem:
    """Manages shared observation transformation and multiple agents"""

    def __init__(self, map_size: Tuple[int, int] = (128, 128),
                 grid_size: int = 128,
                 unit_data_file: str = None, training_conf: Dict = None):
        self.training_conf = training_conf or {}
        self.shared_transformer = SharedObservationTransformer(
            map_size=map_size,
            unit_data_file=unit_data_file,
            training_conf=self.training_conf
        )
        self.unit_agents = {}
        self.current_observation_id = 0
        self.grid_size = grid_size

    def create_unit_agent(self, agent_id: str, cnn_input_size: Tuple[int, int] = (128, 128), dilation_size: int = 3) -> UnitControlAgent:
        arch = self.training_conf.get("agent_architecture", "original")
        
        if arch == "universal":
            from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
            agent = UniversalUnitControlAgent(
                cnn_input_size=cnn_input_size,
                grid_size=self.grid_size,
                dilation_size=dilation_size,
                spatial_resolution=self.training_conf.get("spatial_resolution", 16)
            )
        else:
            agent = UnitControlAgent(
                cnn_input_size=cnn_input_size,
                grid_size=self.grid_size,
                dilation_size=dilation_size
            )
        self.unit_agents[agent_id] = agent
        return agent

    def process_observation(self, observation: Dict[str, Any], player_id: int = 1) -> Dict[str, np.ndarray]:
        """
        Process observation through the hierarchical system
        """
        # Transform observation once
        obs_id = f"obs_{self.current_observation_id}"
        full_transformed = self.shared_transformer.transform_observation(
            observation, player_id=player_id, observation_id=obs_id
        )

        # Extract Game Info for agents (Critical for Coordinate Mapping)
        game_info = observation.get('game_info')

        results = {}
        for agent_id, agent in self.unit_agents.items():
            # Pass game_info so agent can calculate masks and coordinates
            processed = agent.process_transformed_observation(full_transformed, game_info=game_info)
            results[agent_id] = processed

        self.current_observation_id += 1
        return results

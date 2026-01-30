import os
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
from sc2.bot_ai import BotAI
from sc2.main import run_game
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2.position import Point2
from loguru import logger

from src.hierarchical_agent import HierarchicalAgentSystem
from src.ui.debug_ui import DebugUI
from src.ui.ui_manager import UIManager

class MockAgent:
    """A minimal mock of UnitControlAgent to satisfy DebugUI requirements."""
    def __init__(self):
        self.transformed_observation = None
        self.last_action = None
    
    def get_action_mask(self):
        return np.ones(128 * 128, dtype=bool)

from PyQt5.QtCore import QObject, pyqtSignal

class ValidatorSignals(QObject):
    frame_updated = pyqtSignal(np.ndarray, int) # obs, action_idx

class ValidatorBot(BotAI):
    """
    A bot that executes a sequence of recorded actions from a .npz dataset.
    """
    def __init__(self, dataset_path: str, config: Dict[str, Any], mock_agent: MockAgent):
        super().__init__()
        self.signals = ValidatorSignals()
        # ... rest of init ...
        data = np.load(dataset_path)
        self.observations = data["observations"]
        self.actions = data["actions"]
        self.current_step = 0
        self.total_steps = len(self.actions)
        
        self.config = config
        self.mock_agent = mock_agent
        self.action_heads = config.get("env", {}).get("action_heads", ["move"])
        self.grid_size = config.get("env", {}).get("grid_size", 128)

    async def on_step(self, iteration: int):
        if self.current_step >= self.total_steps:
            logger.info("Validation sequence complete.")
            await self.client.leave()
            return

        # 1. Update Mock Agent with recorded observation for DebugUI
        obs = self.observations[self.current_step]
        action_idx = self.actions[self.current_step]
        
        self.mock_agent.transformed_observation = obs
        self.mock_agent.last_action = action_idx
        
        # 2. Notify UI
        self.signals.frame_updated.emit(obs, action_idx)
        
        # 3. Extract and execute action
        await self._apply_action(action_idx)
        
        self.current_step += 1


    async def _apply_action(self, action: int):
        """Logic similar to SC2Bot.apply_action but simplified for validation."""
        grid_area = self.grid_size * self.grid_size
        head_idx = action // grid_area
        spatial_idx = action % grid_area
        
        if head_idx >= len(self.action_heads):
            return

        head_name = self.action_heads[head_idx]
        
        # Convert spatial index to Point2
        gy, gx = divmod(spatial_idx, self.grid_size)
        # Simple mapping (assuming map is matched to grid size for validation)
        target_pos = Point2((gx, gy))
        
        # For validation, we don't know which unit was 'active' in the dataset 
        # (unless we improve the translator to save unit tags).
        # For now, apply to all our units to see the 'ghost' movements.
        for unit in self.units:
            if head_name == "move":
                self.do(unit.move(target_pos))
            elif head_name == "attack":
                self.do(unit.attack(target_pos))

class Validator:
    def __init__(self, project_root: str = "."):
        self.project_root = project_root

    def start_validation(self, dataset_path: str, level_name: str, map_name: str, config: Dict[str, Any]):
        """
        Launches SC2 and DebugUI to visualize the dataset playback.
        """
        mock_agent = MockAgent()
        ui_manager = UIManager(mock_agent)
        
        # Resolve map
        from imitation_learning.recorder import Recorder
        rec = Recorder(self.project_root)
        map_path = rec.get_map_path(level_name, map_name)
        agent_race, enemy_race = rec._parse_races_from_map_name(map_name)

        bot = ValidatorBot(dataset_path, config, mock_agent)
        
        # Launch UI in a separate thread if needed, but for validation we can just show it
        # Note: PyQt needs to run in main thread. This architecture might need refinement
        # depending on how it's called from app.py.
        return bot, map_path, agent_race, enemy_race, ui_manager

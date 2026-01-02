from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import multiprocessing.connection
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.data import Race

from src.hierarchical_agent import HierarchicalAgentSystem

class CaptureTheFlagBot(BotAI):
    """
    Bot for Capture the Flag environment.
    Supports individual unit control and reporting state to the environment.
    """
    def __init__(self, player_id: int, training_conf: Optional[Dict] = None):
        super().__init__()
        self.player_id = player_id
        self.training_conf = training_conf or {}
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        
        # Communication with Env (we don't use conn here, 
        # as MultiClientEnv will likely call methods directly or via a wrapper)
        self.active_unit_tag: Optional[int] = None
        self.command_target: Optional[Point2] = None

    async def on_start(self):
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("active_agent")

    async def on_step(self, iteration: int):
        # The environment will drive the actions.
        # This bot mainly provides the observation.
        pass

    def get_observation(self) -> Dict[str, Any]:
        """Returns the raw observation from this bot's perspective."""
        
        # Determine the target for the objective layer (Enemy Pylon)
        # We need to know where the enemy pylon is.
        # The Env will provide this or we find it in enemy_units.
        enemy_pylon = self.enemy_structures(UnitTypeId.PYLON).first if self.enemy_structures(UnitTypeId.PYLON).exists else None
        target_pos = enemy_pylon.position if enemy_pylon else self.command_target
        
        # Highlighted unit for command_assignment
        selected_units = []
        if self.active_unit_tag:
            active_unit = self.units.find_by_tag(self.active_unit_tag)
            if active_unit:
                selected_units = [active_unit]

        raw_obs = {
            'units': self.units,
            'enemy_units': self.enemy_units,
            'enemy_structures': self.enemy_structures,
            'structures': self.structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': target_pos,
            'selected_units': selected_units
        }
        
        # Process with agent system
        processed = self.agent_system.process_observation(raw_obs)
        return processed["active_agent"]

    async def apply_action(self, unit_tag: int, action: int):
        """Applies a spatial action to a specific unit."""
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return

        world_x, world_y = self.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world(action)
        target = Point2((world_x, world_y))
        
        # For CTF, we usually want Attack Move or Move. 
        # Let's use Move for now as per spec "Agent reached the beacon" logic from Level 0.
        # But for fighting, Attack might be better. 
        # Spec says "Damaging other unit gains a small reward".
        
        self.do(unit.move(target))
        self.command_target = target

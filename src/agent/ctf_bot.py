from typing import Dict, Optional, Any, Tuple, List
import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.data import Race
from loguru import logger

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
        
        self.active_unit_tag: Optional[int] = None
        self.command_target: Optional[Point2] = None
        
        # State tracking for rewards
        self.prev_pylon_health = {1: 20.0, 2: 20.0} # Combined HP + Shield
        self.prev_enemy_units_health = 0.0
        self.prev_own_units_health = 0.0

    async def on_start(self):
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("active_agent")

    def get_observation(self) -> np.ndarray:
        """Returns the processed observation from this bot's perspective."""
        enemy_pylon = self.enemy_structures(UnitTypeId.PYLON).first if self.enemy_structures(UnitTypeId.PYLON).exists else None
        target_pos = enemy_pylon.position if enemy_pylon else self.command_target
        
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
        
        processed = self.agent_system.process_observation(raw_obs)
        return processed["active_agent"]

    async def apply_action(self, unit_tag: int, action: int):
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        world_x, world_y = self.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world(action)
        target = Point2((world_x, world_y))
        
        # CTF uses Attack Move to engage enemies and Pylons
        self.do(unit.attack(target))
        self.command_target = target

    def calculate_reward(self) -> float:
        """
        Calculates CTF rewards:
        - Damaging enemy Pylon: High reward
        - Damage to own Pylon: Negative reward
        - Damaging enemy units: Small reward
        - Taking damage: Small penalty
        - Win/Loss: Max reward/penalty
        """
        reward = 0.0
        r_conf = self.training_conf.get("rewards", {})
        
        # 1. Pylon Health Delta
        p1_pylon = self.structures(UnitTypeId.PYLON).first if self.player_id == 1 else self.enemy_structures(UnitTypeId.PYLON).first
        p2_pylon = self.structures(UnitTypeId.PYLON).first if self.player_id == 2 else self.enemy_structures(UnitTypeId.PYLON).first
        
        def get_hp(pylon):
            if not pylon: return 0.0
            return pylon.health + pylon.shield

        curr_p1_hp = get_hp(p1_pylon) if self.player_id == 1 else get_hp(p2_pylon)
        curr_p2_hp = get_hp(p2_pylon) if self.player_id == 1 else get_hp(p1_pylon) # Enemy pylon from P1 perspective is P2
        
        # Damaging enemy pylon (Enemy is index 2 for P1)
        enemy_idx = 2 if self.player_id == 1 else 1
        own_idx = self.player_id
        
        enemy_pylon_hp = get_hp(self.enemy_structures(UnitTypeId.PYLON).first)
        own_pylon_hp = get_hp(self.structures(UnitTypeId.PYLON).first)
        
        # Pylon Damage Rewards
        if enemy_pylon_hp < self.prev_pylon_health[enemy_idx]:
            reward += (self.prev_pylon_health[enemy_idx] - enemy_pylon_hp) * r_conf.get("pylon_damage_reward", 2.0)
            
        if own_pylon_hp < self.prev_pylon_health[own_idx]:
            reward -= (self.prev_pylon_health[own_idx] - own_pylon_hp) * r_conf.get("pylon_damage_penalty", 2.0)

        self.prev_pylon_health[own_idx] = own_pylon_hp
        self.prev_pylon_health[enemy_idx] = enemy_pylon_hp
        
        # 2. Combat Rewards
        curr_enemy_hp = sum(u.health + u.shield for u in self.enemy_units)
        if curr_enemy_hp < self.prev_enemy_units_health:
            reward += (self.prev_enemy_units_health - curr_enemy_hp) * r_conf.get("damage_reward", 0.1)
        self.prev_enemy_units_health = curr_enemy_hp
        
        curr_own_hp = sum(u.health + u.shield for u in self.units if not u.is_structure)
        if curr_own_hp < self.prev_own_units_health:
            reward -= (self.prev_own_units_health - curr_own_hp) * r_conf.get("damage_penalty", 0.1)
        self.prev_own_units_health = curr_own_hp

        # 3. Win/Loss (Checked in Env, but can be added here as final bonus)
        return reward
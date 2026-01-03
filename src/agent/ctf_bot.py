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
        
        # Satisfaction for BotAI initialization
        self._initialize_variables()
        from sc2.units import Units
        self.units: Units = Units([], self)
        self.structures: Units = Units([], self)
        self.enemy_units: Units = Units([], self)
        self.enemy_structures: Units = Units([], self)
        self.all_units: Units = Units([], self)
        self.larva: Units = Units([], self)
        self.workers: Units = Units([], self)
        self.townhalls: Units = Units([], self)
        self.gas_buildings: Units = Units([], self)
        self.minerals: int = 0
        self.vespene: int = 0
        self.supply_army: int = 0
        self.supply_workers: int = 0
        self.supply_cap: int = 0
        self.supply_used: int = 0
        self.supply_left: int = 0
        self.idle_worker_count: int = 0
        self.army_count: int = 0
        self.warp_gate_count: int = 0
        
        # State tracking for rewards
        self.prev_pylon_health = {1: 20.0, 2: 20.0} # Combined HP + Shield
        self.prev_enemy_units_health = 0.0
        self.prev_own_units_health = 0.0
        
        self.cached_observation: Optional[np.ndarray] = None

    async def on_start(self):
        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            training_conf=self.training_conf
        )
        self.agent_system.create_unit_agent("active_agent")

    def get_observation(self, perspective_player_id: Optional[int] = None) -> np.ndarray:
        """Returns the processed observation from the given player's perspective."""
        # Use provided perspective or default to bot's own player_id
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id
        
        # Defensive check for game info/state
        if not hasattr(self, "game_info") or self.game_info is None:
             logger.warning(f"Bot {self.player_id}: game_info is missing. Returning zero observation.")
             return np.zeros((23, 128, 128), dtype=np.uint8)

        # Flip units based on perspective
        if p_id == self.player_id:
            units = self.units
            enemy_units = self.enemy_units
            structures = self.structures
            enemy_structures = self.enemy_structures
        else:
            # God-Mode: Flip perspectives
            units = self.enemy_units
            enemy_units = self.units
            structures = self.enemy_structures
            enemy_structures = self.structures

        enemy_pylon = enemy_structures(UnitTypeId.PYLON).first if enemy_structures(UnitTypeId.PYLON).exists else None
        target_pos = enemy_pylon.position if enemy_pylon else self.command_target
        
        selected_units = []
        if self.active_unit_tag:
            try:
                # We search in ALL units for the active unit tag
                active_unit = self.all_units.find_by_tag(self.active_unit_tag)
                if active_unit:
                    selected_units = [active_unit]
            except Exception:
                pass

        raw_obs = {
            'units': units,
            'enemy_units': enemy_units,
            'enemy_structures': enemy_structures,
            'structures': structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos,
            'selected_units': selected_units
        }
        
        # Process with agent system, passing p_id for perspective-aware transformation
        try:
            processed = self.agent_system.process_observation(raw_obs, player_id=p_id)
            self.cached_observation = processed["active_agent"]
        except Exception as e:
            if self.cached_observation is not None:
                logger.warning(f"Bot {self.player_id} (P_ID {p_id}): Observation processing failed ({e}). Returning cached observation.")
                return self.cached_observation
            else:
                # Fallback to zeros if nothing cached
                num_layers = self.training_conf.get("active_layers_count", 23)
                return np.zeros((num_layers, 128, 128), dtype=np.uint8)

        return self.cached_observation

    async def apply_action(self, unit_tag: int, action: int):
        unit = self.all_units.find_by_tag(unit_tag)
        if not unit: return

        # Restore dual-head logic parity with Level 7
        grid_area = 128 * 128
        is_attack = False
        spatial_idx = action
        
        if action >= grid_area:
            is_attack = True
            spatial_idx = action - grid_area

        world_x, world_y = self.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world(spatial_idx)
        target = Point2((world_x, world_y))
        
        if is_attack:
            self.do(unit.attack(target))
        else:
            self.do(unit.move(target))
            
        self.command_target = target

        def calculate_reward(self, perspective_player_id: Optional[int] = None) -> float:

            """

            Calculates CTF rewards from the given player's perspective.

            """

            p_id = perspective_player_id if perspective_player_id is not None else self.player_id

            

            reward = 0.0

            r_conf = self.training_conf.get("rewards", {})

            

            # In God-Mode (Single Client), self.player_id is always 1.

            # structures = P1, enemy_structures = P2

            

            p1_pylon_units = self.structures(UnitTypeId.PYLON)

            p2_pylon_units = self.enemy_structures(UnitTypeId.PYLON)

            

            def get_hp(pylon):

                if not pylon: return 0.0

                return pylon.health + pylon.shield

    

            p1_pylon_hp = get_hp(p1_pylon_units.first) if p1_pylon_units.exists else 0.0

            p2_pylon_hp = get_hp(p2_pylon_units.first) if p2_pylon_units.exists else 0.0

            

            # Map indices to HP

            curr_pylon_hp = {1: p1_pylon_hp, 2: p2_pylon_hp}

            

            own_idx = p_id

            enemy_idx = 2 if p_id == 1 else 1

            

            # Pylon Damage Rewards

            if curr_pylon_hp[enemy_idx] < self.prev_pylon_health[enemy_idx]:

                reward += (self.prev_pylon_health[enemy_idx] - curr_pylon_hp[enemy_idx]) * r_conf.get("pylon_damage_reward", 2.0)

                

            if curr_pylon_hp[own_idx] < self.prev_pylon_health[own_idx]:

                reward -= (self.prev_pylon_health[own_idx] - curr_pylon_hp[own_idx]) * r_conf.get("pylon_damage_penalty", 2.0)

    

            # Update previous HP

            self.prev_pylon_health[1] = p1_pylon_hp

            self.prev_pylon_health[2] = p2_pylon_hp

            

            # 2. Combat Rewards

            own_units = self.units if p_id == 1 else self.enemy_units

            enemy_units = self.enemy_units if p_id == 1 else self.units

            

            curr_enemy_hp = sum(u.health + u.shield for u in enemy_units if not u.is_structure)

            if curr_enemy_hp < self.prev_enemy_units_health:

                reward += (self.prev_enemy_units_health - curr_enemy_hp) * r_conf.get("damage_reward", 0.1)

            self.prev_enemy_units_health = curr_enemy_hp

            

            curr_own_hp = sum(u.health + u.shield for u in own_units if not u.is_structure)

            if curr_own_hp < self.prev_own_units_health:

                reward -= (self.prev_own_units_health - curr_own_hp) * r_conf.get("damage_penalty", 0.1)

            self.prev_own_units_health = curr_own_hp

    

            return reward

    
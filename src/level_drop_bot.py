from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from loguru import logger
from typing import Tuple, Dict, List
import numpy as np

class LevelDropBot(Level7Bot):
    """
    Specialized Bot for Drop Operations Training.
    Focus: Loading, Moving, Unloading.
    """
    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.previous_cargo_sum = 0
        self.previous_units_in_zone = set()
        self.command_center_tag = None

    async def reset_game_state(self):
        """
        No spawning logic needed! The map is pre-placed.
        We just need to find our bearings.
        """
        self.previous_cargo_sum = 0
        self.previous_units_in_zone = set()
        self.previous_distances = {}
        self.current_assignments = {}
        self.previous_health = {}
        
        # Reset objective tracking
        self.command_center_tag = None
        self._objective_pos = None
        
        # Initial attempt to identify Enemy Main Structure (Objective)
        # We don't warn here because it might not be visible at frame 0
        self._find_objective()

    def _find_objective(self):
        """
        Scans for the primary objective (enemy main structure).
        """
        main_structures = self.enemy_structures.filter(lambda s: s.type_id in {
            UnitTypeId.COMMANDCENTER, UnitTypeId.HATCHERY, UnitTypeId.NEXUS,
            UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS, UnitTypeId.LAIR, UnitTypeId.HIVE
        })
        
        if main_structures:
            target = main_structures.first
            self.command_center_tag = target.tag
            self._objective_pos = target.position
            logger.info(f"Objective identified: {target.type_id} at {target.position}")
            return True
        return False

    def get_active_units(self):
        """
        Returns active allied units (Marines and Medivacs).
        """
        if not hasattr(self, "units"): return []
        
        # We control both Marines and Medivacs for the drop sequence
        controlled_units = self.units.filter(lambda u: u.type_id in {UnitTypeId.MARINE, UnitTypeId.MEDIVAC})
        return sorted(controlled_units, key=lambda u: u.tag)

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        # Override to use dynamic context and persistent objective
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return np.zeros((1, 1, 1)), {}

        original_unit = self.zergling
        self.zergling = unit # 'zergling' is the centering unit alias
        
        # 1. Lazy Objective Discovery: if missing, try to find it
        if not self.command_center_tag:
            self._find_objective()
            # If still missing after 100 steps (approx), warn once.
            if not self.command_center_tag and self.state.game_loop > 100 and self.state.game_loop % 500 == 0:
                logger.warning(f"Step {self.state.game_loop}: Still no enemy main structure found! Reward logic may be broken.")

        # 2. Persistent Objective: Try to find the structure by tag (persistent across FoW if tag known)
        target_pos = None
        if self.command_center_tag:
            # We first check if the structure is visible
            target = self.enemy_structures.find_by_tag(self.command_center_tag)
            if target:
                target_pos = target.position
            else:
                # If NOT visible, we use the LAST KNOWN position if we have it
                # Or we can look into the full raw structure list if provided by engine?
                # Actually, SC2 persistent objects are usually stored in 'self.structures' 
                # but FoW removes them unless we manually track them.
                
                # For this level, we use 'Intelligence' (A choice in spec):
                # We can iterate over the entire known structure list from the client if needed,
                # but standard BotAI only shows visible ones.
                
                # Better: Cache the position during reset_game_state and reuse it here.
                target_pos = getattr(self, "_objective_pos", None)

        raw_obs = {
            'units': self.units,
            'selected_units': [unit],
            'enemy_units': self.enemy_units,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos
        }
        
        if self.agent_system:
             processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        else:
             processed_obs = np.zeros((1, 64, 64))
             
        self.zergling = original_unit
        return processed_obs, {}

    def calculate_drop_reward(self) -> float:
        total_reward = 0.0
        # Read from env.rewards first, fallback to root rewards (legacy), then default
        env_conf = self.training_conf.get('env', {})
        rewards_conf = env_conf.get('rewards')
        if not rewards_conf:
             rewards_conf = self.training_conf.get('rewards', {})
             
        load_reward = rewards_conf.get('load_reward', 0.1)
        unload_reward = rewards_conf.get('unload_reward', 1.0)
        
        # 1. Load Reward
        # We check all allied units (including flying ones)
        current_cargo_sum = sum(u.cargo_used for u in self.units if hasattr(u, "cargo_used"))
        cargo_diff = current_cargo_sum - self.previous_cargo_sum
        
        if cargo_diff > 0:
            total_reward += cargo_diff * load_reward
            
        self.previous_cargo_sum = current_cargo_sum
        
        # 2. Arrival Reward
        cc = self.structures.find_by_tag(self.command_center_tag)
        if cc:
            # Units on ground near CC
            units_in_zone = {u.tag for u in self.units 
                             if not u.is_flying 
                             and u.distance_to(cc) < 15.0}
            
            new_arrivals = units_in_zone - self.previous_units_in_zone
            if new_arrivals:
                count = len(new_arrivals)
                total_reward += count * unload_reward
                logger.info(f"{count} units arrived at Drop Zone! Reward: {count * unload_reward}")
                
            self.previous_units_in_zone = units_in_zone
            
        return total_reward
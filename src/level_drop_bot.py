from src.level_7_bot import Level7Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from loguru import logger
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
        
        # Identify Command Center (Objective)
        ccs = self.structures(UnitTypeId.COMMANDCENTER)
        if ccs:
            self.command_center_tag = ccs.first.tag
            logger.info(f"Objective identified: CC at {ccs.first.position}")
        else:
            logger.warning("No Command Center found! Reward logic will be broken.")
            self.command_center_tag = None

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
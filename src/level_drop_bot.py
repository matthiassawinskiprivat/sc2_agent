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
        
        # Load Action Heads
        env_conf = self.training_conf.get("env", {})
        self.action_heads = env_conf.get("action_heads", ["move", "attack", "load", "unload"])

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

    async def apply_action(self, unit_tag, action):
        """
        Handles dynamic actions based on configured heads.
        """
        if not self.agent_system: return
        
        grid_area = self.grid_size * self.grid_size
        spatial_idx = action % grid_area
        head_idx = action // grid_area
        
        if head_idx >= len(self.action_heads):
            logger.error(f"Action index {head_idx} out of bounds for heads {self.action_heads}")
            return
            
        action_name = self.action_heads[head_idx]
        
        # Resolve target
        # We need agent_0 to convert spatial to world
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
        target_pos = Point2((world_x, world_y))
        
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        cmd = None
        if action_name == "move":
            cmd = unit.move(target_pos)
        elif action_name == "attack":
            cmd = unit.attack(target_pos)
        elif action_name == "load": # Smart on Unit
            # Find transport at target_pos
            # We search for ANY unit close to target_pos (radius 2.0)
            # Prefer Transports
            candidates = self.units.filter(lambda u: u.distance_to(target_pos) < 3.0)
            transport = next((u for u in candidates if u.cargo_max > 0), None)
            
            if transport:
                cmd = unit.smart(transport)
            else:
                # If no transport, just move there (fallback)
                cmd = unit.move(target_pos)
                
        elif action_name == "unload":
            # Only valid if unit is a transport
            if unit.cargo_used > 0:
                cmd = unit.unload_all_at(target_pos)
        
        if cmd:
            self.do(cmd)

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

    def action_masks(self) -> np.ndarray:
        """
        Dynamically constructs masks based on action_heads configuration.
        """
        # Get base masks from Level7Bot (which currently returns concatenated [Move, Attack])
        # We need to deconstruct this to get the underlying 'pathing' mask vs 'unrestricted' mask.
        
        grid_area = self.grid_size * self.grid_size
        level7_mask = super().action_masks()
        
        # Assumption: Level7Bot returns [Move (Masked), Attack (Unmasked)]
        # We assume the first grid_area is the "Move/Pathing" mask.
        if level7_mask.shape[0] >= grid_area:
            pathing_mask = level7_mask[:grid_area]
        else:
            # Fallback if something is wrong
            pathing_mask = np.ones(grid_area, dtype=bool)
            
        unrestricted_mask = np.ones(grid_area, dtype=bool)
        
        final_masks = []
        for head in self.action_heads:
            if head == "move":
                final_masks.append(pathing_mask)
            elif head == "attack":
                # Attack is usually unrestricted (Attack-Move)
                final_masks.append(unrestricted_mask) 
            elif head == "load":
                # Load is unrestricted (smart command)
                final_masks.append(unrestricted_mask)
            elif head == "unload":
                # Unload is unrestricted
                final_masks.append(unrestricted_mask)
            else:
                logger.warning(f"Unknown action head '{head}', using unrestricted mask.")
                final_masks.append(unrestricted_mask)
                
        return np.concatenate(final_masks)

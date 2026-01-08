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

    async def apply_action(self, unit_tag, action):
        """
        Handles Move (0..N), Attack (N..2N), Load (2N..3N), Unload (3N..4N).
        """
        if not self.agent_system: return
        
        grid_area = self.grid_size * self.grid_size
        spatial_idx = action % grid_area
        action_type = action // grid_area # 0=Move, 1=Attack, 2=Load, 3=Unload
        
        # Resolve target
        # We need agent_0 to convert spatial to world
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
        target_pos = Point2((world_x, world_y))
        
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        cmd = None
        if action_type == 0: # Move
            cmd = unit.move(target_pos)
        elif action_type == 1: # Attack
            cmd = unit.attack(target_pos)
        elif action_type == 2: # Load (Smart on Unit)
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
                
        elif action_type == 3: # Unload
            # Only valid if unit is a transport
            if unit.cargo_used > 0:
                cmd = unit.unload_all_at(target_pos)
        
        if cmd:
            self.do(cmd)

    def calculate_drop_reward(self) -> float:
        total_reward = 0.0
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
        # Override to support 4 Heads (Move, Attack, Load, Unload)
        # Base mask (Move)
        move_mask = super().action_masks() 
        # Level7Bot returns [Move, Attack]. We need to see if that's what we get.
        # Actually Level7Bot.action_masks returns concatenated [Move, Attack].
        
        # For Drop, we want [Move, Attack, Load, Unload]
        # We can construct it by repeating/modifying.
        
        # Get base single-head mask (just pathing)
        # Since Level7Bot combines them, we need to be careful.
        # Let's call Level3Bot (Grandparent) mask? 
        # No, Level3Bot mask is just Move.
        
        # Let's assume Level7Bot gives us [Move, Attack].
        # We will assume Load/Unload are spatially unconstrained (can try to click anywhere).
        # Optimization: Only allow clicking on valid targets? 
        # No, let the neural net learn.
        
        # We need grid_area
        grid_area = self.grid_size * self.grid_size
        
        # Check what super returns
        # If Level7Bot returns 2 * grid, we take the first half as "Move".
        combined_mask = super().action_masks()
        base_move = combined_mask[:grid_area]
        
        # Load Mask: Valid everywhere? Or restricted?
        # Let's make it Valid Everywhere (True) for now.
        load_mask = np.ones(grid_area, dtype=bool)
        
        # Unload Mask: Valid Everywhere
        unload_mask = np.ones(grid_area, dtype=bool)
        
        # Attack Mask (From Level 7)
        attack_mask = combined_mask[grid_area:]
        
        return np.concatenate([base_move, attack_mask, load_mask, unload_mask])

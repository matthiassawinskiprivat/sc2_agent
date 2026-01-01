from src.level_3_bot import Level3Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict, List

class Level7Bot(Level3Bot):
    """
    Level 7 Bot: Combat Basics (Move vs Attack).
    Controls a specific unit type against an enemy unit type.
    """
    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.previous_health = {} # tag -> health (enemy)
        self.previous_allied_health = {} # tag -> health (allied)
        
        # Resolve unit types from configuration
        allied_name = self.training_conf.get("unit_type", "MARINE").upper()
        enemy_name = self.training_conf.get("enemy_type", "ZERGLING").upper()
        
        try:
            self.allied_unit_type = getattr(UnitTypeId, allied_name)
        except AttributeError:
            logger.warning(f"Unknown allied unit type: {allied_name}, falling back to MARINE")
            self.allied_unit_type = UnitTypeId.MARINE
            
        try:
            self.enemy_unit_type = getattr(UnitTypeId, enemy_name)
        except AttributeError:
            logger.warning(f"Unknown enemy unit type: {enemy_name}, falling back to ZERGLING")
            self.enemy_unit_type = UnitTypeId.ZERGLING

    async def reset_game_state(self):
        """
        Spawns Allied and Enemy units based on configuration.
        """
        self.previous_distances = {}
        self.current_assignments = {}
        self.previous_health = {}
        self.previous_allied_health = {}
        
        # Cleanup
        tags_to_kill = []
        # Kill everything
        all_units = self.all_units
        if all_units:
             tags_to_kill.extend([u.tag for u in all_units])
             
        if tags_to_kill:
            await self.client.debug_kill_unit(tags_to_kill)
        
        # Grid Sampling for Spawns
        grid = self.game_info.pathing_grid
        valid_spawn_points = []
        # Safety check for grid bounds
        for x in range(4, grid.width - 4, 4):
            for y in range(4, grid.height - 4, 4):
                if grid[(x, y)] != 0:
                    valid_spawn_points.append(Point2((x, y)))
        
        if len(valid_spawn_points) < self.targets_count * 2:
             # Fallback
             w, h = self.game_info.map_size
             valid_spawn_points = [Point2((w/2, h/2)) for _ in range(self.targets_count * 2)]

        locations = random.sample(valid_spawn_points, self.targets_count * 2)
        
        # Spawn Pairs
        for i in range(self.targets_count):
            start = locations[i]
            end = locations[self.targets_count + i]
            
            # Ally (Player 1)
            await self.client.debug_create_unit([[self.allied_unit_type, 1, start, self.player_id]])
            
            # Enemy (Player 2)
            await self.client.debug_create_unit([[self.enemy_unit_type, 1, end, 2]])

    def get_active_units(self):
        """Returns active allied units."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(self.allied_unit_type), key=lambda u: u.tag)

    async def apply_action(self, unit_tag, action):
        """
        Handles Move (0..N) and Attack (N..2N).
        """
        if not self.agent_system: return
        
        grid_area = self.grid_size * self.grid_size
        is_attack = False
        spatial_idx = action
        
        if action >= grid_area:
            is_attack = True
            spatial_idx = action - grid_area
            
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
        target_pos = Point2((world_x, world_y))
        
        unit = self.units.find_by_tag(unit_tag)
        if unit:
            if is_attack:
                cmd = unit.attack(target_pos)
            else:
                cmd = unit.move(target_pos)
                
            if cmd:
                self.do(cmd)

    def calculate_combat_reward(self) -> float:
        """
        Calculates reward based on damage dealt and kills.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        damage_reward = rewards_conf.get('damage_reward', 0.1)
        kill_reward = rewards_conf.get('kill_reward', 10.0)
        
        total_reward = 0.0
        
        # 1. Track Enemy Health (Damage Dealt)
        enemies = self.enemy_units(self.enemy_unit_type)
        current_enemy_tags = {u.tag for u in enemies}
        
        # Check for damage on existing enemies
        for u in enemies:
            prev = self.previous_health.get(u.tag, u.health_max + u.shield_max)
            curr = u.health + u.shield
            damage = max(0, prev - curr)
            if damage > 0:
                total_reward += damage * damage_reward
            self.previous_health[u.tag] = curr
            
        # Check for Kills
        tracked_tags = set(self.previous_health.keys())
        died_tags = tracked_tags - current_enemy_tags
        
        for tag in died_tags:
             total_reward += kill_reward
             del self.previous_health[tag]
             
        return total_reward

    def calculate_allied_attrition_penalty(self) -> float:
        """
        Calculates penalty based on allied damage taken and units killed.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        damage_penalty = rewards_conf.get('damage_penalty', -0.1)
        kill_penalty = rewards_conf.get('kill_penalty', -10.0)
        
        total_penalty = 0.0
        
        allies = self.units(self.allied_unit_type)
        current_allied_tags = {u.tag for u in allies}
        
        # 1. Track Allied Damage
        for u in allies:
            prev = self.previous_allied_health.get(u.tag, u.health_max + u.shield_max)
            curr = u.health + u.shield
            damage = max(0, prev - curr)
            if damage > 0:
                total_penalty += damage * damage_penalty
            self.previous_allied_health[u.tag] = curr
            
        # 2. Track Allied Kills (Deaths)
        tracked_tags = set(self.previous_allied_health.keys())
        died_tags = tracked_tags - current_allied_tags
        
        for tag in died_tags:
            total_penalty += kill_penalty
            del self.previous_allied_health[tag]
            
        return total_penalty

    def get_assigned_distances_combat(self) -> Dict[int, float]:
        """
        Calculates optimal assignment and returns distances to assigned enemy units.
        """
        self.current_assignments = self._calculate_optimal_assignment_combat()
        distances = {}
        for u_tag, e_tag in self.current_assignments.items():
            unit = self.units.find_by_tag(u_tag)
            enemy = self.enemy_units.find_by_tag(e_tag)
            if unit and enemy:
                distances[u_tag] = unit.distance_to(enemy)
        return distances

    def calculate_distance_reward(self) -> float:
        """
        Calculates the distance-based shaping reward using optimal combat assignments.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        step_penalty = rewards_conf.get('step_penalty', -0.05)
        dist_multiplier = rewards_conf.get('distance_multiplier', 0.1)
        
        current_distances = self.get_assigned_distances_combat()
        total_dist_change = 0.0
        
        for u_tag, dist in current_distances.items():
            prev_dist = self.previous_distances.get(u_tag)
            if prev_dist is not None:
                total_dist_change += (prev_dist - dist)
            
            self.previous_distances[u_tag] = dist
            
        active_count = len(self.get_active_units())
        avg_dist_reward = (total_dist_change * dist_multiplier) / max(1, active_count)
        
        return avg_dist_reward + step_penalty

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        # Override to use dynamic context
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return np.zeros((1, 1, 1)), {}

        original_unit = self.zergling
        self.zergling = unit # 'zergling' is the centering unit alias
        
        # Update assignments
        assignments = self._calculate_optimal_assignment_combat()
        assigned_target_tag = assignments.get(unit.tag)
        
        target_pos = None
        if assigned_target_tag:
            target = self.enemy_units.find_by_tag(assigned_target_tag)
            if target:
                target_pos = target.position

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

    def _calculate_optimal_assignment_combat(self) -> Dict[int, int]:
        """
        Allied -> Enemy assignment.
        """
        allies = self.get_active_units()
        enemies = self.enemy_units(self.enemy_unit_type)
        
        if not allies or not enemies:
            return {}
            
        a_coords = np.array([[u.position.x, u.position.y] for u in allies])
        e_coords = np.array([[u.position.x, u.position.y] for u in enemies])
        
        from src.utils.model_utils import solve_optimal_assignment
        a_idx, e_idx = solve_optimal_assignment(a_coords, e_coords)
        
        assignment = {}
        for i, j in zip(a_idx, e_idx):
             assignment[allies[i].tag] = enemies[j].tag
        return assignment


    def action_masks(self) -> np.ndarray:
        """
        Returns a concatenated mask for Move (first half) and Attack (second half).
        """
        # Get the base move mask from the agent system
        move_mask = super().action_masks()
        
        # For Level 7, we want 2 * GridSize^2
        # Move mask (first half) is restricted by pathing.
        # Attack mask (second half) is unmasked to allow Attack-Move to any location.
        attack_mask = np.ones_like(move_mask, dtype=bool)
        
        return np.concatenate([move_mask, attack_mask])

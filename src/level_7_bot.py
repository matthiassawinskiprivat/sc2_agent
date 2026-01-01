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
    Controls Marines against Zerglings.
    """
    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.previous_health = {} # tag -> health

    async def reset_game_state(self):
        """
        Spawns Marines (Ally) and Zerglings (Enemy).
        """
        self.previous_distances = {}
        self.current_assignments = {}
        self.previous_health = {}
        
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
            
            # Ally: Marine (Player 1)
            await self.client.debug_create_unit([[UnitTypeId.MARINE, 1, start, self.player_id]])
            
            # Enemy: Zergling (Player 2)
            # Note: Player 2 ID depends on how the game is set up. Usually 2.
            await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, end, 2]])

    def get_active_units(self):
        """Returns active Marines."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(UnitTypeId.MARINE), key=lambda u: u.tag)

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
        enemies = self.enemy_units(UnitTypeId.ZERGLING)
        current_enemy_tags = {u.tag for u in enemies}
        
        # Check for damage on existing enemies
        for u in enemies:
            prev = self.previous_health.get(u.tag, u.health_max + u.shield_max)
            curr = u.health + u.shield
            damage = max(0, prev - curr)
            if damage > 0:
                total_reward += damage * damage_reward
            self.previous_health[u.tag] = curr
            
        # Check for Kills (Enemies present before but missing now)
        # Note: This requires previous_health to track enemies across frames
        # Since Level 7 has NO Fog of War, missing = dead.
        
        tracked_tags = set(self.previous_health.keys())
        
        # Identify died tags (present in previous but not current)
        died_tags = tracked_tags - current_enemy_tags
        
        for tag in died_tags:
             total_reward += kill_reward
             del self.previous_health[tag]
             
        return total_reward

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        # Override to use Marine/Zergling context
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return np.zeros((1, 1, 1)), {}

        original_zergling = self.zergling
        self.zergling = unit # 'zergling' property is just 'controlled unit' alias in base class
        
        # Update assignments (Marine -> Zergling)
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
            'enemy_units': self.enemy_units, # Pass actual enemies
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
             
        self.zergling = original_zergling
        return processed_obs, {}

    def _calculate_optimal_assignment_combat(self) -> Dict[int, int]:
        """
        Marine -> Zergling assignment.
        """
        marines = self.get_active_units()
        zerglings = self.enemy_units(UnitTypeId.ZERGLING)
        
        if not marines or not zerglings:
            return {}
            
        m_coords = np.array([[u.position.x, u.position.y] for u in marines])
        z_coords = np.array([[u.position.x, u.position.y] for u in zerglings])
        
        # Reuse the solver from model_utils (imported in Level3Bot)
        from src.utils.model_utils import solve_optimal_assignment
        m_idx, z_idx = solve_optimal_assignment(m_coords, z_coords)
        
        assignment = {}
        for i, j in zip(m_idx, z_idx):
             assignment[marines[i].tag] = zerglings[j].tag
        return assignment

    def action_masks(self) -> np.ndarray:
        """
        Returns a concatenated mask for Move (first half) and Attack (second half).
        """
        # Get the base move mask from the agent system
        move_mask = super().action_masks()
        
        # For Level 7, we want 2 * GridSize^2
        # We use the same move_mask for both Move and Attack for now,
        # as pathing is the primary constraint.
        return np.concatenate([move_mask, move_mask])

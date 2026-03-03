from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict, List
from scipy.optimize import linear_sum_assignment


class Level3Bot(SC2Bot):
    """
    Level 3 Bot for handling multiple units and multiple targets (beacons).
    It implements a cycle-based control where each unit gets an observation
    and takes an action sequentially within a single game step.
    """
    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.targets_count = self.training_conf.get("targets", 1)
        
        # Dynamic unit types for Level 3
        # Look in training_conf root first, then in 'env' section
        env_conf = self.training_conf.get("env", {})
        allied_name = self.training_conf.get("unit_type", env_conf.get("unit_type", "ZERGLING")).upper()
        try:
            self.allied_unit_type = getattr(UnitTypeId, allied_name)
        except AttributeError:
            self.allied_unit_type = UnitTypeId.ZERGLING
            
        self.beacon_type = UnitTypeId.BEACON_TERRANSMALL
        
        self.active_zerglings = [] 
        self.active_beacons = []
        
        # previous_distances is used to calculate reward based on progress towards beacons.
        self.previous_distances = {} # Map unit_tag -> distance to assigned beacon
        self.current_assignments = {} # Map unit_tag -> beacon_tag
        self.pending_success_rewards = {} # unit_tag -> float (buffered rewards)
        self.completed_tags = set() # tags (units/beacons) that should be removed

    async def reset_game_state(self, verify: bool = True):
        """
        Cleans up the map and spawns new units and Beacons.
        """
        await super().reset_game_state(verify=verify)
        self.previous_distances = {}
        self.current_assignments = {}
        self.pending_success_rewards = {}
        self.completed_tags = set()
        
        # Spawn new units at valid pathable locations
        grid = self.game_info.pathing_grid
        valid_spawn_points = []
        for x in range(4, grid.width - 4, 4):
            for y in range(4, grid.height - 4, 4):
                if grid[(x, y)] != 0:
                    valid_spawn_points.append(Point2((x, y)))
        
        if len(valid_spawn_points) < self.targets_count * 2:
            w, h = self.game_info.map_size
            for _ in range(self.targets_count):
                start = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
                end = Point2((random.uniform(w*0.1, w*0.9), random.uniform(h*0.1, h*0.9)))
                await self.client.debug_create_unit([[self.allied_unit_type, 1, start, self.player_id]])
                await self.client.debug_create_unit([[self.beacon_type, 1, end, self.player_id]])
        else:
            locations = random.sample(valid_spawn_points, self.targets_count * 2)
            for i in range(self.targets_count):
                start = locations[i]
                end = locations[self.targets_count + i]
                await self.client.debug_create_unit([[self.allied_unit_type, 1, start, self.player_id]])
                await self.client.debug_create_unit([[self.beacon_type, 1, end, self.player_id]])

        # Step game to actually spawn units and refresh state
        await self.client.step()
        await self._refresh_state()
        
        # Initialize optimal persistent assignments at the start of the episode
        self.current_assignments = self.get_optimal_assignments()

    def get_optimal_assignments(self) -> Dict[int, int]:
        """
        Calculates the optimal unit-to-beacon assignment using the Hungarian algorithm
        (linear sum assignment) to minimize total distance.
        Handles surplus units by assigning them to their closest beacon.
        """
        units = self.get_active_zerglings()
        beacons = self.all_units(self.beacon_type)
        
        if not units or not beacons.exists:
            return {}
            
        num_units = len(units)
        num_beacons = beacons.amount
        
        # Create distance matrix [unit_idx, beacon_idx]
        dist_matrix = np.zeros((num_units, num_beacons))
        beacon_list = list(beacons)
        
        for i, unit in enumerate(units):
            for j, beacon in enumerate(beacon_list):
                dist_matrix[i, j] = unit.distance_to(beacon)
                
        # Solve linear sum assignment to minimize total distance
        # Handles non-square matrices (min(units, beacons) assignments)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        
        assignment = {}
        for r, c in zip(row_ind, col_ind):
            # Map unit tag to beacon tag
            assignment[units[r].tag] = beacon_list[c].tag
            
        # SURPLUS HANDLING: Ensure EVERY unit has a target if targets exist.
        if num_units > num_beacons:
            for unit in units:
                if unit.tag not in assignment:
                    # Assign surplus unit to its closest beacon (many-to-one)
                    closest_beacon = beacons.closest_to(unit)
                    assignment[unit.tag] = closest_beacon.tag
            
        return assignment

    def get_closest_targets(self) -> Dict[int, int]:
        """
        Calculates the greedy unit-to-beacon assignment where each unit
        targets its closest beacon.
        """
        units = self.get_active_zerglings()
        beacons = self.all_units(self.beacon_type)
        
        if not units or not beacons.exists:
            return {}
            
        assignment = {}
        for unit in units:
            # Defensive check: ensure beacons still exists before calling closest_to
            if beacons.exists:
                closest_beacon = beacons.closest_to(unit)
                assignment[unit.tag] = closest_beacon.tag
            
        return assignment

    def get_assigned_distances(self) -> Dict[int, float]:
        """
        Returns distances to the currently assigned beacons.
        If assignments are missing (e.g. at start), calculates them optimally.
        """
        if not self.current_assignments:
            self.current_assignments = self.get_optimal_assignments()
            
        distances = {}
        for u_tag, b_tag in self.current_assignments.items():
            unit = self.units.find_by_tag(u_tag)
            beacon = self.all_units.find_by_tag(b_tag)
            if unit and beacon:
                distances[u_tag] = unit.distance_to(beacon)
        return distances

    def calculate_distance_reward(self, unit_tag: int = None) -> float:
        """
        Calculates the distance-based shaping reward for a specific unit or all units.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        dist_multiplier = rewards_conf.get('distance_multiplier', 0.1)
        
        current_distances = self.get_assigned_distances()
        
        if unit_tag is not None:
            dist = current_distances.get(unit_tag)
            if dist is None:
                return 0.0
                
            prev_dist = self.previous_distances.get(unit_tag)
            reward = 0.0
            if prev_dist is not None:
                reward = (prev_dist - dist) * dist_multiplier
            
            self.previous_distances[unit_tag] = dist
            return reward

        # Fallback: calculate total change for all units
        total_dist_change = 0.0
        for u_tag, dist in current_distances.items():
            prev_dist = self.previous_distances.get(u_tag)
            if prev_dist is not None:
                total_dist_change += (prev_dist - dist)
            self.previous_distances[u_tag] = dist
            
        return total_dist_change * dist_multiplier

    def get_active_units(self):
        """Returns sorted list of active units for Level 3."""
        return self.get_active_zerglings()

    def action_masks(self, curriculum_progress: float = 1.0) -> np.ndarray:
        """
        Overrides action_masks to ensure the guidance mask uses the 
        CORRECT assigned beacon for the currently active unit.
        """
        # 1. Identify which unit is active (using the same logic as SC2Bot.action_masks)
        active_unit_tag = getattr(self, "active_unit_tag", None)
        active_unit = None
        if active_unit_tag:
            active_unit = self.all_units.find_by_tag(active_unit_tag)
        
        if not active_unit:
            # Fallback to index-based for legacy support
            active_units = self.get_active_units()
            # Level3Env should keep self.current_unit_index in sync on the bot
            idx = getattr(self, "current_unit_index", 0)
            if 0 <= idx < len(active_units):
                active_unit = active_units[idx]
            
        # 2. Set self.beacon to the assigned beacon for this unit
        if active_unit:
            if not self.current_assignments:
                self.current_assignments = self.get_optimal_assignments()
            
            b_tag = self.current_assignments.get(active_unit.tag)
            if b_tag:
                self.beacon = self.all_units.find_by_tag(b_tag)
        
        # 3. Call base implementation which now uses the correctly set self.beacon
        return super().action_masks(curriculum_progress=curriculum_progress)

    def get_active_zerglings(self):
        """Returns sorted list of active units."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(self.allied_unit_type), key=lambda u: u.tag)

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        """
        Generates centered observation for specific unit.
        """
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return np.zeros((1, 1, 1)), {} 
            
        return self._get_observation(active_unit=unit)
        
    def _get_observation(self, active_unit: Unit = None) -> Tuple[np.ndarray, Dict]:
        """
        Override to implement selective targeting and handle dynamic re-assignment.
        """
        if active_unit is None:
            return super()._get_observation()
            
        # 1. Dynamic Re-assignment: If the assigned beacon is gone, pick a new one
        if not self.current_assignments:
            self.current_assignments = self.get_optimal_assignments()
            
        assigned_beacon_tag = self.current_assignments.get(active_unit.tag)
        target_beacon = self.all_units.find_by_tag(assigned_beacon_tag) if assigned_beacon_tag else None

        # If the target is missing but beacons still exist, re-calculate
        if not target_beacon and self.all_units(self.beacon_type).exists:
            self.current_assignments = self.get_optimal_assignments()
            assigned_beacon_tag = self.current_assignments.get(active_unit.tag)
            target_beacon = self.all_units.find_by_tag(assigned_beacon_tag) if assigned_beacon_tag else None

        # 2. Layer Rendering Parity
        # Show ONLY the assigned beacon to the agent on the COMMAND_OBJECTIVE layer.
        # This helps the model focus on its specific target in multi-unit scenarios.
        all_beacons = list(self.all_units(self.beacon_type))
        
        # ALLIED UNITS: Should NOT include any beacons.
        allied_units = self.units.filter(lambda u: u.type_id != self.beacon_type)

        raw_obs = {
            'units': allied_units,
            'selected_units': [active_unit],
            'enemy_units': all_beacons, # All beacons remain visible in enemy layer
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            # Focus COMMAND_OBJECTIVE on the specific assigned target
            'command_target': [target_beacon] if target_beacon else []
        }
        
        if self.agent_system:
             processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        else:
             processed_obs = np.zeros((1, 64, 64))
             
        return processed_obs, {}

    def get_global_observation(self, active_unit: Unit = None) -> np.ndarray:
        """
        Generates a GLOBAL observation that includes all units and beacons.
        Useful for UI rendering where we want to see the full state.
        """
        all_beacons = list(self.all_units(self.beacon_type))
        allied_units = self.units.filter(lambda u: u.type_id != self.beacon_type)
        
        # For the global view, we show the objective of the currently active unit
        target_beacon = None
        target_unit = active_unit
        if not target_unit:
            active_units = self.get_active_zerglings()
            if active_units:
                target_unit = active_units[0]
                
        if target_unit:
            b_tag = self.current_assignments.get(target_unit.tag)
            if b_tag:
                target_beacon = self.all_units.find_by_tag(b_tag)

        raw_obs = {
            'units': allied_units,
            'selected_units': [target_unit] if target_unit else [],
            'enemy_units': all_beacons, # All beacons are visible
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': target_beacon
        }
        
        if self.agent_system:
             return self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        return np.zeros((1, 64, 64))
        
    async def check_collections(self) -> Dict[int, float]:
        """
        Checks for unit-beacon collisions for ONLY assigned pairs.
        This enforces that units follow their assigned objectives.
        Returns ONLY the rewards for units that JUST reached their beacon this step.
        """
        unit_rewards = {}
        
        # Ensure current_assignments exist
        if not self.current_assignments:
            self.current_assignments = self.get_optimal_assignments()
            
        if not self.current_assignments:
            return {}
        
        rewards_conf = self.training_conf.get('rewards', {})
        success_reward_total = rewards_conf.get('success', 100.0)
        success_reward_per_event = success_reward_total / max(1, self.targets_count)
        
        # We iterate over a copy of items to allow deletion
        for u_tag, b_tag in list(self.current_assignments.items()):
            # Skip if already completed but not yet removed from game
            if u_tag in self.completed_tags:
                continue

            unit = self.units.find_by_tag(u_tag)
            beacon = self.all_units.find_by_tag(b_tag)
            
            if unit and beacon:
                # Robust distance check
                try:
                    dist = unit.distance_to(beacon)
                    if not isinstance(dist, (int, float, np.number)):
                         dist = ((unit.position.x - beacon.position.x)**2 + (unit.position.y - beacon.position.y)**2)**0.5
                except:
                    dist = 999.0
                
                if dist < 2.0:
                    logger.info(f"Collection! Unit {u_tag} reached assigned Beacon {b_tag}")
                    # Mark for deletion
                    self.completed_tags.add(u_tag)
                    self.completed_tags.add(b_tag)
                    
                    # Buffer reward
                    self.pending_success_rewards[u_tag] = self.pending_success_rewards.get(u_tag, 0.0) + success_reward_per_event
                    unit_rewards[u_tag] = success_reward_per_event
                    
                    # Cleanup assignments targeting these entities
                    # We do NOT delete from current_assignments yet, to keep the 'target' visible
                    # until the unit is actually killed in its turn.
            elif not unit or not beacon:
                 # Clean up assignments if units/beacons are gone for other reasons
                 if u_tag in self.current_assignments:
                     del self.current_assignments[u_tag]
        
        return unit_rewards


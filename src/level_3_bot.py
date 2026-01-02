from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict, List


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
        allied_name = self.training_conf.get("unit_type", "ZERGLING").upper()
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

    async def reset_game_state(self, verify: bool = True):
        """
        Cleans up the map and spawns new units and Beacons.
        """
        await super().reset_game_state(verify=verify)
        self.previous_distances = {}
        self.current_assignments = {}
        
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

    def get_closest_targets(self) -> Dict[int, int]:
        """
        Calculates the greedy unit-to-beacon assignment where each unit
        targets its closest beacon.
        """
        units = self.get_active_zerglings()
        beacons = self.units(self.beacon_type)
        
        if not units or not beacons:
            return {}
            
        assignment = {}
        for unit in units:
            closest_beacon = beacons.closest_to(unit)
            assignment[unit.tag] = closest_beacon.tag
            
        return assignment

    def get_assigned_distances(self) -> Dict[int, float]:
        """
        Calculates greedy assignment and returns distances to closest beacons.
        """
        self.current_assignments = self.get_closest_targets()
        distances = {}
        for u_tag, b_tag in self.current_assignments.items():
            unit = self.units.find_by_tag(u_tag)
            beacon = self.units.find_by_tag(b_tag)
            if unit and beacon:
                distances[u_tag] = unit.distance_to(beacon)
        return distances

    def calculate_distance_reward(self) -> float:
        """
        Calculates the distance-based shaping reward using optimal assignments.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        step_penalty = rewards_conf.get('step_penalty', -0.05)
        dist_multiplier = rewards_conf.get('distance_multiplier', 0.1)
        
        current_distances = self.get_assigned_distances()
        total_dist_change = 0.0
        
        for u_tag, dist in current_distances.items():
            prev_dist = self.previous_distances.get(u_tag)
            if prev_dist is not None:
                total_dist_change += (prev_dist - dist)
            
            self.previous_distances[u_tag] = dist
            
        active_count = len(self.get_active_zerglings())
        avg_dist_reward = (total_dist_change * dist_multiplier) / max(1, active_count)
        
        return avg_dist_reward + step_penalty

    def get_active_zerglings(self):
        """Returns sorted list of active units."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(self.allied_unit_type), key=lambda u: u.tag)

    async def apply_action(self, unit_tag, action):
        """
        Translates spatial action to world-space move command.
        """
        if not self.agent_system: return
        
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(action)
        target_pos = Point2((world_x, world_y))
        
        unit = self.units.find_by_tag(unit_tag)
        if unit:
            cmd = unit.move(target_pos)
            if cmd:
                self.do(cmd)

    def get_observation_for_unit(self, unit_tag) -> Tuple[np.ndarray, Dict]:
        """
        Generates centered observation for specific unit.
        """
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return np.zeros((1, 1, 1)), {} 
            
        original_unit = self.zergling
        self.zergling = unit
        
        beacons = self.units(self.beacon_type)
        if beacons.exists:
            self.beacon = beacons.closest_to(unit)
        else:
            self.beacon = None
            
        assignments = self.get_closest_targets()
        assigned_beacon_tag = assignments.get(unit.tag)
        
        target_pos = None
        if assigned_beacon_tag:
            assigned_beacon = self.units.find_by_tag(assigned_beacon_tag)
            if assigned_beacon:
                target_pos = assigned_beacon.position

        raw_obs = {
            'units': self.units,
            'selected_units': [unit],
            'enemy_units': list(beacons) if beacons.exists else [],
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
        
    def _get_observation(self) -> Tuple[np.ndarray, Dict]:
        return super()._get_observation()
        
    async def check_collections(self):
        """
        Checks for unit-beacon collisions.
        """
        reward = 0.0
        
        units = self.units(self.allied_unit_type)
        beacons = self.units(self.beacon_type)
        
        if not units.exists or not beacons.exists:
            return 0.0
        
        rewards_conf = self.training_conf.get('rewards', {})
        success_reward_total = rewards_conf.get('success', 100.0)
        success_reward_per_event = success_reward_total / max(1, self.targets_count)
        
        tags_to_kill = []
        collected_units = set()
        collected_beacons = set()
        
        for u in units:
            if u.tag in collected_units:
                continue
            for b in beacons:
                if b.tag in collected_beacons:
                    continue
                
                dist = u.distance_to(b)
                if dist < 2.0:
                    logger.info(f"Collection! Unit {u.tag} reached Beacon {b.tag}")
                    tags_to_kill.append(u.tag)
                    tags_to_kill.append(b.tag)
                    collected_units.add(u.tag)
                    collected_beacons.add(b.tag)
                    reward += success_reward_per_event
                    break 
        
        if tags_to_kill:
            await self.client.debug_kill_unit(tags_to_kill)
            
        return reward

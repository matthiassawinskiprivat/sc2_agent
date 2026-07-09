from src.sc2_bot import SC2Bot
from src.sc2_bot_universal import SC2BotUniversal
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict, List, Any
from scipy.optimize import linear_sum_assignment


class Level3Bot(SC2Bot):
    """
    Level 3 Bot for handling multiple units and multiple targets (beacons).
    It implements a cycle-based control where each unit gets an observation
    and takes an action sequentially within a single game step.
    """
    def action_masks(self, curriculum_progress: float = 1.0) -> Any:
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
            # Universal MultiDiscrete masking
            return SC2BotUniversal.action_masks(self, curriculum_progress)
        
        # ORIGINAL FLAT DISCRETE SPACE MASKING
        # 1. Identify which unit is active
        active_unit_tag = getattr(self, "active_unit_tag", None)
        active_unit = None
        if active_unit_tag:
            active_unit = self.all_units.find_by_tag(active_unit_tag)
        
        if not active_unit:
            active_units = self.get_active_units()
            idx = getattr(self, "current_unit_index", 0)
            if 0 <= idx < len(active_units):
                active_unit = active_units[idx]
            
        # 2. Set self.beacon to the assigned beacon for this unit
        if active_unit:
            b_tag = self.current_assignments.get(active_unit.tag) if self.current_assignments else None
            if b_tag:
                self.beacon = self.all_units.find_by_tag(b_tag)
        
        # 3. Call base implementation which now uses the correctly set self.beacon
        return super().action_masks(curriculum_progress=curriculum_progress)

    async def apply_action(self, unit_tag, action):
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
            return await SC2BotUniversal.apply_action(self, unit_tag, action)
        return await super().apply_action(unit_tag, action)

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
        # Filled once per episode in reset_game_state; pairings are never recomputed mid-episode.
        self.current_assignments = {} # Map unit_tag -> beacon_tag
        self.pending_success_rewards = {} # unit_tag -> float (buffered rewards)
        self.completed_tags = set() # tags (units/beacons) that should be removed

    def _spawn_geometry_key(self) -> tuple:
        """Identity for pathing/playable geometry; invalidates spawn cache on map rotation."""
        area = self.game_info.playable_area
        grid = self.game_info.pathing_grid
        ms = self.game_info.map_size
        if hasattr(ms, "width"):
            mw, mh = int(ms.width), int(ms.height)
        else:
            mw, mh = int(ms[0]), int(ms[1])
        return (
            mw,
            mh,
            int(area.x),
            int(area.y),
            int(area.width),
            int(area.height),
            int(getattr(grid, "width", 0)),
            int(getattr(grid, "height", 0)),
        )

    def _rebuild_spawn_point_cache(self) -> List[Point2]:
        grid = self.game_info.pathing_grid
        area = self.game_info.playable_area
        margin = 5.0
        # Intersect pathing grid scan with playable margin (128 edge safety) without
        # shrinking below the legacy full-grid scan when playable covers the map.
        min_x = max(4, int(area.x + margin))
        max_x = min(int(getattr(grid, "width", 0)) - 4, int(area.x + area.width - margin))
        min_y = max(4, int(area.y + margin))
        max_y = min(int(getattr(grid, "height", 0)) - 4, int(area.y + area.height - margin))
        return [
            Point2((x, y))
            for x in range(min_x, max_x, 4)
            for y in range(min_y, max_y, 4)
            if grid[(x, y)] != 0
        ]

    def _random_pathable_in_playable(self, max_tries: int = 50) -> Point2:
        grid = self.game_info.pathing_grid
        area = self.game_info.playable_area
        margin = 5.0
        for _ in range(max_tries):
            x = random.uniform(float(area.x + margin), float(area.x + area.width - margin))
            y = random.uniform(float(area.y + margin), float(area.y + area.height - margin))
            try:
                if grid[(int(x), int(y))] != 0:
                    return Point2((x, y))
            except Exception:
                pass
        return Point2((float(area.x + area.width / 2), float(area.y + area.height / 2)))

    def unit_is_level3_scenario_cleanup(self, u) -> bool:
        """
        Only remove units this level spawns.

        1v1 ladder maps pre-place a full enemy base (Alliance.Enemy). The default
        SC2Bot cleanup treats every non-neutral unit as killable (~150 on typical
        AIE maps) even when no Computer opponent is in the game.
        """
        if getattr(u, "owner_id", None) != getattr(self, "player_id", 1):
            return False
        return u.type_id in (self.allied_unit_type, self.beacon_type)

    async def reset_game_state(self, verify: bool = True):
        """
        Cleans up the map and spawns new units and Beacons.
        """
        # Ladder maps often retain map-placed units that never fully clear via
        # debug_kill within a few steps; we still run cleanup but must not abort
        # the worker when debris remains (see SC2Bot.reset_game_state).
        await super().reset_game_state(
            verify=verify,
            strict_empty_map=False,
            cleanup_predicate=self.unit_is_level3_scenario_cleanup,
        )
        self.previous_distances = {}
        self.current_assignments = {}
        self.pending_success_rewards = {}
        self.completed_tags = set()
        
        # Spawn new units at valid pathable locations inside playable_area.
        # Cache spawn points per map geometry (hard reset rotates ladder maps).
        geom_key = self._spawn_geometry_key()
        if getattr(self, "_spawn_cache_key", None) != geom_key or not getattr(self, "_cached_spawn_points", None):
            self._cached_spawn_points = self._rebuild_spawn_point_cache()
            self._spawn_cache_key = geom_key
        valid_spawn_points = self._cached_spawn_points
        
        # Build all spawn commands as one batched call instead of 2*targets_count
        # individual awaited SC2 RPC calls.  Each individual call has localhost
        # websocket round-trip overhead; batching reduces this from 8 calls to 1.
        spawn_cmds = []
        if len(valid_spawn_points) < self.targets_count * 2:
            for _ in range(self.targets_count):
                start = self._random_pathable_in_playable()
                end = self._random_pathable_in_playable()
                spawn_cmds.append([self.allied_unit_type, 1, start, self.player_id])
                spawn_cmds.append([self.beacon_type,      1, end,   self.player_id])
        else:
            locations = random.sample(valid_spawn_points, self.targets_count * 2)
            for i in range(self.targets_count):
                spawn_cmds.append([self.allied_unit_type, 1, locations[i],                      self.player_id])
                spawn_cmds.append([self.beacon_type,      1, locations[self.targets_count + i], self.player_id])
        await self.client.debug_create_unit(spawn_cmds)

        # Step game to actually spawn units and refresh state
        await self.client.step()
        await self._refresh_state()
        
        # Initialize optimal persistent assignments at the start of the episode
        self.current_assignments = self.get_optimal_assignments()

        units = self.get_active_zerglings()
        beacons = self.all_units(self.beacon_type)
        if len(units) != self.targets_count or beacons.amount != self.targets_count:
            raise RuntimeError(
                f"Level3 spawn verification failed: expected {self.targets_count} "
                f"{self.allied_unit_type.name} and {self.targets_count} {self.beacon_type.name}, "
                f"got {len(units)} and {beacons.amount} (map may have conflicting pre-placed units)."
            )

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
        assigned_beacons = set()
        for r, c in zip(row_ind, col_ind):
            # Map unit tag to beacon tag
            u_tag = units[r].tag
            b_tag = beacon_list[c].tag
            assignment[u_tag] = b_tag
            assigned_beacons.add(b_tag)
            
        # SURPLUS HANDLING: Ensure EVERY unit has a target if targets exist.
        if num_units > num_beacons:
            for unit in units:
                if unit.tag not in assignment:
                    # Assign surplus unit to its closest beacon (many-to-one)
                    closest_beacon = beacons.closest_to(unit)
                    assignment[unit.tag] = closest_beacon.tag
        
        # TARGET SURPLUS HANDLING: Ensure EVERY beacon has at least one unit assigned.
        # This prevents "orphaned" beacons if some units die.
        if num_beacons > num_units:
            unassigned_beacons = [b for b in beacon_list if b.tag not in assigned_beacons]
            for beacon in unassigned_beacons:
                # Assign this beacon to its closest unit (many-to-one)
                closest_unit = units[0] # Fallback
                min_dist = float('inf')
                for unit in units:
                    d = unit.distance_to(beacon)
                    if d < min_dist:
                        min_dist = d
                        closest_unit = unit
                
                # We can only store ONE assignment per unit in the current Dict structure.
                # However, Level 3 cycle logic processes one unit at a time.
                # To handle multiple beacons per unit, we would need a list.
                # BUT: The environment only asks for ONE action per unit.
                # DECISION: We stick to 1-target-per-unit, but re-calculate if target is gone.
                # Since we already have dynamic re-assignment in _get_observation, 
                # any unit whose target is collected will automatically pick up one of the 
                # remaining (previously "unassigned") beacons in the next step.
                pass
            
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

    def _find_allied_unit_by_tag(self, unit_tag: int):
        """Resolve player-controlled unit; prefer `units`, fall back to `all_units`."""
        if unit_tag is None or not hasattr(self, "units"):
            return None
        u = self.units.find_by_tag(unit_tag)
        if u is not None:
            return u
        if hasattr(self, "all_units"):
            return self.all_units.find_by_tag(unit_tag)
        return None

    def get_assigned_distances(self) -> Dict[int, float]:
        """
        Returns distances to the currently assigned beacons.
        If assignments are missing (e.g. at start), calculates them optimally.
        """
        distances = {}
        for u_tag, b_tag in self.current_assignments.items():
            unit = self._find_allied_unit_by_tag(u_tag)
            beacon = self.all_units.find_by_tag(b_tag)
            if unit and beacon:
                distances[u_tag] = unit.distance_to(beacon)
        return distances

    def calculate_distance_reward(self, unit_tag: int = None) -> float:
        """
        Calculates the distance-based shaping reward for a specific unit or all units.
        """
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
             # SC2BotUniversal uses a simpler single-unit reward based on self.zergling/beacon
             # We need to bridge this for Level 3's multi-unit attribution.
             if unit_tag is not None:
                 unit = self._find_allied_unit_by_tag(unit_tag)
                 # Find assigned beacon
                 b_tag = self.current_assignments.get(unit_tag)
                 beacon = self.all_units.find_by_tag(b_tag) if b_tag else None
                 
                 if not unit or not beacon:
                     return 0.0
                     
                 current_dist = unit.distance_to(beacon)
                 prev_dist = self.previous_distances.get(unit_tag)
                 
                 reward = 0.0
                 if prev_dist is not None:
                     dist_multiplier = self.training_conf.get('rewards', {}).get('distance_multiplier', 1.0)
                     reward = (prev_dist - current_dist) * dist_multiplier
                     
                 self.previous_distances[unit_tag] = current_dist
                 return reward
             return 0.0

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

    def get_active_zerglings(self):
        """Returns sorted list of active units."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(self.allied_unit_type), key=lambda u: u.tag)

    def _get_zero_observation(self) -> np.ndarray:
        """Returns a zero observation matching the current architecture's shape."""
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
            # Return Dict for Universal
            num_layers = len(self.training_conf.get("active_layers", [])) or 30 # Fallback
            tactical_layers = len(self.training_conf.get("active_tactical_layers", [])) or num_layers
            return {
                "map": np.zeros((num_layers, 128, 128), dtype=np.float32),
                "tactical": np.zeros((tactical_layers, 32, 32), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32)
            }
        # Return Array for Original
        return np.zeros((1, 64, 64), dtype=np.float32)

    def get_observation_for_unit(self, unit_tag) -> Tuple[Any, Dict]:
        """
        Generates centered observation for specific unit.
        """
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return self._get_zero_observation(), {} 
            
        return self._get_observation(active_unit=unit)
        
    def _get_observation(self, active_unit: Unit = None) -> Tuple[Any, Dict]:
        """
        Override to implement selective targeting. Assignments are computed once per episode
        in reset_game_state and never recomputed; if this unit's beacon no longer exists,
        only that unit's pairing is dropped (no Hungarian re-run on remaining units).
        """
        if active_unit is None:
            # Fallback to base SC2Bot if no unit specified
            if not self.agent_system:
                return self._get_zero_observation(), {}
            return super()._get_observation()

        assigned_beacon_tag = (
            self.current_assignments.get(active_unit.tag) if self.current_assignments else None
        )
        target_beacon = self.all_units.find_by_tag(assigned_beacon_tag) if assigned_beacon_tag else None

        # Stale pairing (beacon collected/killed); do not reshuffle other units' assignments.
        if assigned_beacon_tag and not target_beacon:
            self.current_assignments.pop(active_unit.tag, None)
            self.previous_distances.pop(active_unit.tag, None)
            target_beacon = None

        # Rendering
        # - Enemy density layer should render ALL beacons (training/UI parity).
        # - Command objective layer should render ONLY this unit's assigned beacon via `command_target`.
        all_beacons = list(self.all_units(self.beacon_type))
        
        # ALLIED UNITS: Should NOT include any beacons.
        allied_units = self.units.filter(lambda u: u.type_id != self.beacon_type)

        raw_obs = {
            'units': allied_units,
            'selected_units': [active_unit],
            'enemy_units': all_beacons,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'high_level_command': self.training_conf.get('high_level_command', 0),
            # Pass the Unit object so the transformer can detect it is a BEACON and render it
            # at full strength (val=20.0) in the COMMAND_OBJECTIVE layer.  Passing only
            # target_beacon.position causes the beacon type-check to miss and the signal
            # drops to val=1.0 — a 20× weaker target cue that severely hinders learning.
            'command_target': [target_beacon] if target_beacon else []
        }
        
        if self.agent_system:
             processed_obs = self.agent_system.process_observation(observation=raw_obs, active_unit=active_unit)["agent_0"]
        else:
             processed_obs = self._get_zero_observation()
             
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
            'enemy_units': all_beacons,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'high_level_command': self.training_conf.get('high_level_command', 0),
            'command_target': [target_beacon] if target_beacon else [],
        }
        
        if self.agent_system:
             return self.agent_system.process_observation(observation=raw_obs, active_unit=target_unit)["agent_0"]
        return np.zeros((1, 64, 64))
        
    async def check_collections(self) -> Dict[int, float]:
        """
        Checks for unit-beacon collisions for ONLY assigned pairs.
        This enforces that units follow their assigned objectives.
        Returns ONLY the rewards for units that JUST reached their beacon this step.
        """
        unit_rewards = {}

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

            unit = self._find_allied_unit_by_tag(u_tag)
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
                    # logger.info(f"Collection! Unit {u_tag} reached assigned Beacon {b_tag} (dist: {dist:.2f})")
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
                 # Map triggers often delete the beacon (or unit) immediately upon collision.
                 # If they suddenly disappeared, check if they were close in the last frame.
                 prev_dist = self.previous_distances.get(u_tag, 999.0)

                 if prev_dist < 4.0: # Threshold accounts for Zergling movement speed per frame
                     logger.info(f"Collection inferred (Map Trigger)! Unit {u_tag} reached assigned Beacon {b_tag} (prev_dist: {prev_dist:.2f})")
                     self.completed_tags.add(u_tag)
                     self.completed_tags.add(b_tag)
                     
                     self.pending_success_rewards[u_tag] = self.pending_success_rewards.get(u_tag, 0.0) + success_reward_per_event
                     unit_rewards[u_tag] = success_reward_per_event
                     
                 # Clean up assignments if units/beacons are gone
                 if u_tag in self.current_assignments:
                     del self.current_assignments[u_tag]
        
        return unit_rewards

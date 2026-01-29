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
    Supports Terran (Medivac/Marine), Zerg (Overlord/Zergling), and Protoss (WarpPrism/Zealot).
    """
    TRANSPORT_TYPES = {
        UnitTypeId.MEDIVAC, 
        UnitTypeId.OVERLORD, 
        UnitTypeId.OVERLORDTRANSPORT,
        UnitTypeId.WARPPRISM,
        UnitTypeId.WARPPRISMPHASING
    }

    INFANTRY_TYPES = {
        UnitTypeId.MARINE, UnitTypeId.MARAUDER,
        UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK,
        UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT
    }

    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.previous_cargo_sum = 0
        self.previous_units_in_zone = set()
        self.previous_transport_distances = {}
        self.previous_meeting_distances = {} # tag -> distance to closest pickup partner or centroid
        self.previous_transport_cargo = {} # tag -> cargo_used
        self.previous_passenger_tags = set() # tags of units currently in cargo
        self.initially_loaded_units = set() # tags of units that have been loaded once
        self.target_unloaded_units = set() # tags of units that have been unloaded in target zone once
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
        self.previous_allied_health = {}
        self.previous_transport_distances = {}
        self.previous_meeting_distances = {}
        self.previous_transport_cargo = {}
        self.previous_passenger_tags = set()
        self.initially_loaded_units = set()
        self.target_unloaded_units = set()
        
        # Reset objective tracking
        self.command_center_tag = None
        self._objective_pos = None
        
        # Initial attempt to identify Enemy Main Structure (Objective)
        self._find_objective()
        
        # Initialize tracking for existing entities
        for u in self.enemy_units + self.enemy_structures:
            self.previous_health[u.tag] = u.health + u.shield
        
        all_allies = self.units.filter(lambda u: u.type_id in self.INFANTRY_TYPES or u.type_id in self.TRANSPORT_TYPES)
        for u in all_allies:
            self.previous_allied_health[u.tag] = u.health + u.shield
            if u.type_id in self.TRANSPORT_TYPES:
                self.previous_transport_cargo[u.tag] = u.cargo_used
                if hasattr(u, "passengers"):
                    for p in u.passengers:
                        self.previous_passenger_tags.add(p.tag)
                        self.previous_allied_health[p.tag] = p.health + p.shield
        
        # Initial meeting distances (to centroid if multiple, else to closest)
        self.update_reward_states()

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

    def is_unit_damaged(self, unit) -> bool:
        """
        Returns True if the unit's total health (health + shield) is below 50% of its maximum.
        """
        current_total = unit.health + unit.shield
        max_total = unit.health_max + unit.shield_max
        if max_total == 0: return False
        return (current_total / max_total) < 0.5

    def get_active_units(self):
        """
        Returns active allied units (Infantry and Transports).
        """
        if not hasattr(self, "units"): return []
        
        # We control both Infantry and Transports for the drop sequence
        controlled_units = self.units.filter(lambda u: u.type_id in self.INFANTRY_TYPES or u.type_id in self.TRANSPORT_TYPES)
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

    def calculate_combat_reward(self) -> float:
        """
        Specialized combat reward for Drop Operations.
        Tracks damage to ALL enemy entities (units and structures).
        """
        rewards_conf = self.training_conf.get('rewards', {})
        damage_reward = rewards_conf.get('damage_reward', 0.1)
        kill_reward = rewards_conf.get('kill_reward', 10.0)
        
        total_reward = 0.0
        
        # 1. Track ALL Enemy Health (Damage Dealt)
        all_enemies = self.enemy_units + self.enemy_structures
        current_enemy_tags = {u.tag for u in all_enemies}
        
        for u in all_enemies:
            # If unit is new, initialize its health in tracking WITHOUT giving reward
            if u.tag not in self.previous_health:
                self.previous_health[u.tag] = u.health + u.shield
                continue

            prev = self.previous_health[u.tag]
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
        Specialized attrition penalty for Drop operations.
        Includes passengers in the 'living' set to avoid 'fake death' penalties on load.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        damage_penalty = rewards_conf.get('damage_penalty', -0.1)
        kill_penalty = rewards_conf.get('kill_penalty', -10.0)
        
        total_penalty = 0.0
        
        allies = self.units.filter(lambda u: u.type_id in self.INFANTRY_TYPES or u.type_id in self.TRANSPORT_TYPES)
        living_tags = {u.tag for u in allies}
        all_living_units = list(allies)
        
        for transport in allies.filter(lambda u: u.type_id in self.TRANSPORT_TYPES):
            if hasattr(transport, "passengers"):
                for passenger in transport.passengers:
                    living_tags.add(passenger.tag)
                    all_living_units.append(passenger)
        
        for u in all_living_units:
            # If unit is new (e.g. just spawned or discovered), initialize tracking
            if u.tag not in self.previous_allied_health:
                self.previous_allied_health[u.tag] = u.health + u.shield
                continue

            prev = self.previous_allied_health[u.tag]
            curr = u.health + u.shield
            damage = max(0, prev - curr)
            if damage > 0:
                total_penalty += damage * damage_penalty
                self.previous_allied_health[u.tag] = curr
            
        tracked_tags = set(self.previous_allied_health.keys())
        died_tags = tracked_tags - living_tags
        
        for tag in died_tags:
            if tag in self.previous_allied_health:
                total_penalty += kill_penalty
                del self.previous_allied_health[tag]
            
        return total_penalty

    def calculate_distance_reward(self, unit_tag: int = None) -> float:
        """
        In Drop operations, distance reward is handled by calculate_drop_reward.
        """
        return 0.0

    def calculate_drop_reward(self, active_tag: int = None) -> float:
        total_reward = 0.0
        env_conf = self.training_conf.get('env', {})
        rewards_conf = env_conf.get('rewards')
        if not rewards_conf:
             rewards_conf = self.training_conf.get('rewards', {})
             
        load_reward = rewards_conf.get('load_reward', 0.1)
        unload_reward = rewards_conf.get('unload_reward', 2.0) # Increased from 1.0 to 2.0
        # Support user's preferred naming 'distance_multiplier' while keeping backward compatibility
        distance_reward_scale = rewards_conf.get('distance_multiplier', rewards_conf.get('distance_reward_scale', 0.05))
        meeting_reward_scale = distance_reward_scale * 0.5
        occupancy_penalty = -0.01 # Penalty per unit per step in cargo

        active_unit_found = False
        active_unit_type = None
        is_passenger = False
        acting_info = ""
        
        # Identify acting unit status
        unit = self.all_units.find_by_tag(active_tag)
        if unit:
            active_unit_found = True
            active_unit_type = unit.type_id
            acting_info = f" [Actor: {unit.type_id.name} ({active_tag})]"
        else:
            # Check if it's a passenger
            for transport in self.units.filter(lambda u: u.type_id in self.TRANSPORT_TYPES):
                for passenger in getattr(transport, "passengers", []):
                    if passenger.tag == active_tag:
                        active_unit_found = True
                        active_unit_type = getattr(passenger, "unit_id", None)
                        is_passenger = True
                        acting_info = f" [Actor: Passenger ({active_tag})]"
                        break
                if active_unit_found: break
        
        if not active_unit_found:
            return 0.0

        # --- PHASE 1: PICKUP GUIDANCE (Centroid-based) ---
        if not is_passenger:
            # Transports move to Infantry Group Centroid
            if active_unit_type in self.TRANSPORT_TYPES and unit.cargo_used == 0:
                target_centroid = getattr(self, "_infantry_centroid", None)
                if target_centroid:
                    dist = unit.distance_to(target_centroid)
                    prev_dist = self.previous_meeting_distances.get(active_tag, dist)
                    dist_diff = prev_dist - dist
                    # shaping_reward = dist_diff * meeting_reward_scale
                    # shaping_reward = max(-0.5, min(0.5, shaping_reward))
                    # total_reward += shaping_reward

            # Infantry move to Transport Group Centroid
            elif active_unit_type in self.INFANTRY_TYPES:
                target_centroid = getattr(self, "_transport_centroid", None)
                if target_centroid:
                    dist = unit.distance_to(target_centroid)
                    prev_dist = self.previous_meeting_distances.get(active_tag, dist)
                    dist_diff = prev_dist - dist
                    #if dist > 2.0: # Only reward until close
                        # shaping_reward = dist_diff * meeting_reward_scale
                        # shaping_reward = max(-0.5, min(0.5, shaping_reward))
                        # total_reward += shaping_reward

        # --- PHASE 2: LOADING REWARD (Refined) ---
        if is_passenger and active_tag not in self.previous_passenger_tags:
            # Passenger-side reward (if enabled/needed, currently redundant with transport-side)
            pass 

        if active_unit_type in self.TRANSPORT_TYPES:
            transport = self.units.find_by_tag(active_tag)
            if transport:
                prev_cargo = self.previous_transport_cargo.get(active_tag, 0)
                curr_cargo = transport.cargo_used
                if curr_cargo > prev_cargo:
                    # Identify newly loaded units
                    if hasattr(transport, "passengers"):
                        for p in transport.passengers:
                            if p.tag not in self.previous_passenger_tags:
                                # This unit was just loaded
                                if p.tag not in self.initially_loaded_units:
                                    # Initial Load: Scaled Reward
                                    multiplier = 1.0 + (prev_cargo * 0.25)
                                    unit_reward = load_reward * multiplier
                                    total_reward += unit_reward
                                    self.initially_loaded_units.add(p.tag)
                                    logger.info(f"Initial Load Reward: +{unit_reward:.3f} (Cargo: {prev_cargo}) {acting_info}")
                                elif self.is_unit_damaged(p):
                                    # Emergency Pickup: Flat Reward
                                    total_reward += load_reward
                                    logger.info(f"Emergency Pickup Reward: +{load_reward:.3f} (Damaged) {acting_info}")
                    
                    self.previous_transport_cargo[active_tag] = curr_cargo

        # Occupancy Penalty: Discourage units staying in cargo forever
        if is_passenger:
            total_reward += occupancy_penalty

        # --- PHASE 3: OBJECTIVE GUIDANCE ---
        target_pos = self._objective_pos
        if not target_pos and self.command_center_tag:
            cc = self.enemy_structures.find_by_tag(self.command_center_tag)
            if cc:
                target_pos = cc.position

        # Distance Shaping Reward (Configurable)
        # Can be disabled (set to 0.0) for levels where flanking is required.
        if target_pos and active_unit_type in self.TRANSPORT_TYPES:
            transport = self.units.find_by_tag(active_tag)
            if transport and transport.cargo_used > 0:
                dist = transport.distance_to(target_pos)
                prev_dist = self.previous_transport_distances.get(transport.tag, dist)
                dist_diff = prev_dist - dist
                shaping_reward = dist_diff * distance_reward_scale * transport.cargo_used
                shaping_reward = max(-1.0, min(1.0, shaping_reward))
                total_reward += shaping_reward

        # --- PHASE 4: ARRIVAL REWARD (Refined) ---
        if target_pos and active_unit_type in self.TRANSPORT_TYPES:
            units_in_zone = {u.tag for u in self.units 
                             if not u.is_flying 
                             and u.distance_to(target_pos) < 15.0}
            new_arrivals = units_in_zone - self.previous_units_in_zone
            if new_arrivals:
                # Identify which units were just unloaded from THIS transport
                # (Intersection with previous_passenger_tags ensures they were actually in cargo)
                actually_unloaded = new_arrivals.intersection(self.previous_passenger_tags)
                
                prev_cargo = self.previous_transport_cargo.get(active_tag, 0)
                arrival_reward = 0.0
                unloaded_count = 0
                
                # Sort arrivals to ensure deterministic reward if multiple (though rare in one step)
                for tag in sorted(actually_unloaded):
                    if tag not in self.target_unloaded_units:
                        # Scaled Reward: Highest when transport was full
                        # Formula: R * (1 + (Cargo_Before - 1) * 0.25)
                        # We decrement simulated cargo for each unit in this batch
                        effective_cargo = max(1, prev_cargo - unloaded_count)
                        multiplier = 1.0 + ((effective_cargo - 1) * 0.25)
                        unit_reward = unload_reward * multiplier
                        
                        arrival_reward += unit_reward
                        self.target_unloaded_units.add(tag)
                        unloaded_count += 1
                        logger.info(f"Unit Arrival Reward: +{unit_reward:.3f} (Cargo before: {effective_cargo}) {acting_info}")

                total_reward += arrival_reward
                self.previous_units_in_zone.update(new_arrivals)
                if unloaded_count > 0:
                    logger.info(f"{unloaded_count} units successfully dropped in Zone! Total Reward: {arrival_reward:.3f}{acting_info}")
                
        return total_reward

    def update_reward_states(self):
        """
        Updates tracking states for rewards at the end of each unit cycle.
        Uses Centroids for stable meeting guidance.
        """
        all_allies = self.units.filter(lambda u: u.type_id in self.INFANTRY_TYPES or u.type_id in self.TRANSPORT_TYPES)
        transports = all_allies.filter(lambda u: u.type_id in self.TRANSPORT_TYPES)
        infantry = all_allies.filter(lambda u: u.type_id in self.INFANTRY_TYPES)

        # Calculate Centroids
        if infantry:
            mean_pos = np.mean([(u.position.x, u.position.y) for u in infantry], axis=0)
            self._infantry_centroid = Point2((mean_pos[0], mean_pos[1]))
        else:
            self._infantry_centroid = None

        if transports:
            mean_pos = np.mean([(u.position.x, u.position.y) for u in transports], axis=0)
            self._transport_centroid = Point2((mean_pos[0], mean_pos[1]))
        else:
            self._transport_centroid = None

        # 1. Update Meeting Distances
        for t in transports:
            if self._infantry_centroid:
                self.previous_meeting_distances[t.tag] = t.distance_to(self._infantry_centroid)
        for i in infantry:
            if self._transport_centroid:
                self.previous_meeting_distances[i.tag] = i.distance_to(self._transport_centroid)

        # 2. Update Transport Cargo
        for transport in transports:
            self.previous_transport_cargo[transport.tag] = transport.cargo_used

        # 3. Update Passenger tags
        self.previous_passenger_tags = set()
        for transport in transports:
            if hasattr(transport, "passengers"):
                for p in transport.passengers:
                    self.previous_passenger_tags.add(p.tag)

        # 4. Update Transport Distances to Objective
        target_pos = self._objective_pos
        if target_pos:
            for transport in transports:
                self.previous_transport_distances[transport.tag] = transport.distance_to(target_pos)

            # 5. Update Units in Zone
            self.previous_units_in_zone = {u.tag for u in self.units 
                                         if not u.is_flying 
                                         and u.distance_to(target_pos) < 15.0}

        # 6. Health tracking
        for u in all_allies:
            self.previous_allied_health[u.tag] = u.health + u.shield
            if hasattr(u, "passengers"):
                for p in u.passengers:
                    self.previous_allied_health[p.tag] = p.health + p.shield
        for u in self.enemy_units + self.enemy_structures:
            self.previous_health[u.tag] = u.health + u.shield

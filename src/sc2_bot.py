"""
A simple bot that acts as a wrapper for the HierarchicalAgentSystem,
and handles the communication with the main process.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import multiprocessing
import multiprocessing.connection
import random
from loguru import logger

from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from src.hierarchical_agent import HierarchicalAgentSystem


class SC2Bot(BotAI):
    """A simple bot that wraps the HierarchicalAgentSystem."""

    def __init__(self, conn: multiprocessing.connection.Connection, training_conf: Optional[Dict] = None):
        super().__init__()
        self.conn = conn
        self.training_conf = training_conf or {}
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        self.zergling: Optional[Unit] = None
        self.beacon: Optional[Unit] = None
        self.previous_distance = None
        self.last_target_pos: Optional[Point2] = None
        
        # Load Action Heads
        env_conf = self.training_conf.get("env", {})
        self.action_heads = env_conf.get("action_heads", self.training_conf.get("action_heads", ["move"]))
        self.categorical_heads = env_conf.get("categorical_heads", self.training_conf.get("categorical_heads", []))

        # enemy_race must be set to avoid AttributeError in BotAIInternal._prepare_step
        # when it checks for Race.Random to update it from observations.
        self._enemy_race = Race.Random
        
        # Satisfaction for BotAI initialization
        self._initialize_variables()
        from sc2.units import Units
        self.units = Units([], self)
        self.structures = Units([], self)
        self.enemy_units = Units([], self)
        self.enemy_structures = Units([], self)
        self.all_units = Units([], self)
        
        self.unit_command_uses_self_do = False
        self.actions = []
        self.unit_tags_received_action = set()

    async def on_step(self, iteration: int):
        if iteration == 0:
            # Try to get grid_size from parent env if possible
            # Default to 128, but strictly enforce 16 for level_0
            grid_size = 128
            level_name = getattr(self, "level_name", "")
            if not level_name and hasattr(self, "training_conf"):
                level_name = self.training_conf.get("level_name", "")

            if level_name == "level_0":
                grid_size = 32
            elif hasattr(self, "_bot_object") and hasattr(self._bot_object, "grid_size"):
                grid_size = self._bot_object.grid_size
            elif hasattr(self, "grid_size"):
                grid_size = self.grid_size

            self.agent_system = HierarchicalAgentSystem(
                map_size=(128, 128),
                grid_size=grid_size,
                training_conf=self.training_conf
            )
            self.agent_system.create_unit_agent("agent_0")

            await self.reset_game_state()
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        command, data = self.conn.recv()
        if command == "step":
            action = data
            # Use zergling as default unit for legacy on_step loop
            if self.zergling:
                await self.apply_action(self.zergling.tag, action)
            
            obs, reward, done, truncated, info = self._get_step_results()
            self.conn.send((obs, reward, done, truncated, info))

        elif command == "reset":
            await self.reset_game_state()
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        elif command == "close":
            await self.client.leave()

    async def apply_action(self, unit_tag, action):
        """
        Handles hybrid actions (Spatial + Categorical).
        """
        if not self.agent_system: return
        
        grid_area = self.grid_size * self.grid_size
        total_spatial_indices = len(self.action_heads) * grid_area
        
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        cmd = None

        if action < total_spatial_indices:
            # --- SPATIAL ACTION ---
            head_idx = action // grid_area
            spatial_idx = action % grid_area
            action_name = self.action_heads[head_idx]
            
            # Resolve target
            world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
            target_pos = Point2((world_x, world_y))
            self.last_target_pos = target_pos 
            
            if action_name == "move":
                cmd = unit.move(target_pos)
            elif action_name == "attack":
                cmd = unit.attack(target_pos)
            elif action_name == "load":
                # Map of transport types to their specific LOAD ability (for picking up units).
                transport_load_mapping = {
                    UnitTypeId.MEDIVAC: AbilityId.LOAD_MEDIVAC,
                    UnitTypeId.OVERLORD: AbilityId.LOAD_OVERLORD,
                    UnitTypeId.WARPPRISM: AbilityId.LOAD_WARPPRISM,
                    UnitTypeId.WARPPRISMPHASING: AbilityId.LOAD_WARPPRISM,
                }

                candidates = self.units.filter(lambda u: u.distance_to(target_pos) < 3.0)
                
                if unit.type_id in transport_load_mapping:
                    loadable = next((u for u in candidates if u.cargo_max == 0), None)
                    if loadable:
                        ability = transport_load_mapping[unit.type_id]
                        cmd = unit(ability, loadable)
                    else:
                        cmd = unit.move(target_pos)
                else:
                    transport = next((u for u in candidates if u.cargo_max > 0), None)
                    if transport:
                        cmd = unit.smart(transport)
                    else:
                        cmd = unit.move(target_pos)
            else:
                logger.error(f"Unhandled spatial action head: {action_name}")

        else:
            # --- CATEGORICAL ACTION ---
            cat_idx = action - total_spatial_indices
            if cat_idx >= len(self.categorical_heads):
                logger.error(f"Categorical index {cat_idx} out of bounds")
                return
                
            action_name = self.categorical_heads[cat_idx]
            
            if action_name == "unload":
                # CATEGORICAL UNLOAD (UnloadAll at current position)
                logger.debug(f"Attempting Categorical UNLOAD for {unit.type_id} ({unit.tag}). Cargo: {unit.cargo_used}/{unit.cargo_max}")
                if unit.cargo_used > 0:
                    # Specialized UNLOADALL mapping for reliability
                    mapping = {
                        UnitTypeId.MEDIVAC: AbilityId.UNLOADALLAT_MEDIVAC,
                        UnitTypeId.OVERLORD: AbilityId.UNLOADALLAT_OVERLORD,
                        UnitTypeId.OVERLORDTRANSPORT: AbilityId.UNLOADALLAT_OVERLORD,
                        UnitTypeId.WARPPRISM: AbilityId.UNLOADALLAT_WARPPRISM,
                        UnitTypeId.WARPPRISMPHASING: AbilityId.UNLOADALL_WARPPRISM,
                    }
                    ability = mapping.get(unit.type_id, AbilityId.UNLOADALL)
                    
                    logger.debug(f"Executing categorical UNLOAD for {unit.type_id} ({unit.tag}) using {ability}")
                    
                    # Handle 'AT' variants which require a target position
                    if ability in {AbilityId.UNLOADALLAT_MEDIVAC, AbilityId.UNLOADALLAT_OVERLORD, AbilityId.UNLOADALLAT_WARPPRISM}:
                        cmd = unit(ability, unit.position)
                    else:
                        cmd = unit(ability)
                else:
                    logger.warning(f"Categorical UNLOAD skipped: Unit {unit.tag} has no cargo.")

        if cmd:
            self.do(cmd)

    async def reset_game_state(self, verify: bool = True):
        """
        Base reset logic: Guarantees vision and clears all units from the map.
        """
        self.previous_distance = None
        self.unit_tags_received_action.clear()
        
        # 1. Guarantee Vision
        # We step after debug_show_map to ensure it's applied before we look for units to kill
        await self.client.debug_show_map()
        await self.client.step()
        await self._refresh_state()

        # 2. Robust Cleanup Loop
        # We keep killing until the map is actually empty
        max_cleanup_attempts = 5
        for attempt in range(max_cleanup_attempts):
            all_units = self.all_units
            if not all_units:
                break
                
            logger.info(f"Cleanup attempt {attempt+1}: Killing {all_units.amount} units.")
            await self.client.debug_kill_unit([u.tag for u in all_units])
            await self.client.step()
            await self._refresh_state()
            
            if not self.all_units:
                break
        
        if self.all_units:
            logger.error(f"Failed to clear map after {max_cleanup_attempts} cleanup attempts. Remaining: {self.all_units.amount}")

    def get_active_units(self):
        """Returns active allied units. Default implementation for single-unit bots."""
        if self.zergling:
            return [self.zergling]
        return []

    async def _refresh_state(self):
        """Refreshes the bot's internal state from a new observation."""
        obs = await self.client.observation()
        from sc2.game_state import GameState
        from train.envs.level_0_env import ProtoWrapper
        # Use existing game_info._proto if possible
        proto = getattr(self.game_info, "_proto", None)
        if not proto:
            # Fallback to fetching fresh game info if proto is missing
            game_info = await self.client.get_game_info()
            proto = game_info._proto
            
        self._prepare_step(GameState(obs.observation), ProtoWrapper(proto))

    def action_masks(self) -> np.ndarray:
        if not self.agent_system or "agent_0" not in self.agent_system.unit_agents:
             # Determine fallback grid_size
             gs = 128
             if hasattr(self, "grid_size"):
                 gs = self.grid_size
             elif self.agent_system:
                 gs = self.agent_system.grid_size
             return np.ones(gs * gs, dtype=bool)

        move_mask = self.agent_system.unit_agents["agent_0"].get_action_mask()
        
        # Robust Unit Identification
        active_unit_tag = getattr(self, "active_unit_tag", None)
        active_unit = None
        if active_unit_tag:
            active_unit = self.all_units.find_by_tag(active_unit_tag)
        
        if not active_unit:
            # Fallback to index-based for legacy support
            active_units = self.get_active_units()
            idx = getattr(self, "current_unit_index", 0)
            if 0 <= idx < len(active_units):
                active_unit = active_units[idx]

        is_transport_actor = False
        is_flying = False
        cargo_used = 0

        if active_unit:
            is_flying = active_unit.is_flying
            if is_flying:
                move_mask = np.ones_like(move_mask, dtype=bool)
            
            # Context for Load/Unload heads
            is_transport_actor = active_unit.cargo_max > 0
            cargo_used = getattr(active_unit, "cargo_used", 0)

        unrestricted_mask = np.ones_like(move_mask, dtype=bool)
        zero_mask = np.zeros_like(move_mask, dtype=bool)
        
        # 0. Retrieve Dynamic Ranges (Load/Heal)
        load_range = 4.0
        heal_range = 4.0
        if hasattr(self, "game_data") and hasattr(self.game_data, "abilities"):
            try:
                # Use Medivac Load and Heal as standard reference
                # Some abilities (like LOAD) might return 0.0 in cast_range if they are proximity-based or triggered by smart()
                l_rng = self.game_data.abilities[AbilityId.LOAD_MEDIVAC.value]._proto.cast_range
                h_rng = self.game_data.abilities[AbilityId.MEDIVACHEAL_HEAL.value]._proto.cast_range
                
                # If the game returns 0.0, we use a sensible default (4.0) to allow the agent to function.
                if l_rng > 0: load_range = l_rng
                if h_rng > 0: heal_range = h_rng
            except:
                pass

        final_masks = []
        for head in self.action_heads:
            if head == "move":
                final_masks.append(move_mask)
            elif head == "attack":
                # Special Case: Medivac 'Attack' is actually 'Heal'
                if active_unit and active_unit.type_id == UnitTypeId.MEDIVAC:
                    active_radius = getattr(active_unit, "radius", 0.0)
                    
                    # Find any damaged biological allied units in range
                    # We filter our own units
                    damaged_allies = self.units.filter(lambda u: u.tag != active_unit.tag and u.is_biological and u.health < u.health_max)
                    
                    can_heal = False
                    for u in damaged_allies:
                        target_radius = getattr(u, "radius", 0.0)
                        if active_unit.distance_to(u) <= (heal_range + active_radius + target_radius):
                            can_heal = True
                            break
                    
                    if not can_heal:
                        # Allow move_mask (attempting to heal) even if we don't strictly see a target.
                        # This prevents the agent from freezing if it disagrees with the game engine on range.
                        final_masks.append(move_mask)
                    else:
                        # If can heal, use move_mask (which is already adjusted for flyers/ground in the code above)
                        final_masks.append(move_mask)
                else:
                    # Use move_mask to enforce terrain limits for ground units (move_mask is already adjusted for flyers)
                    final_masks.append(move_mask)
            elif head == "load":
                # Non-transport flyers cannot load
                if is_flying and not is_transport_actor:
                    final_masks.append(zero_mask)
                else:
                    # Use specialized transport mask for the 'load' head
                    game_info = getattr(self, "game_info", getattr(self, "_game_info", None))
                    # Pass is_transport context so Medivacs see Marines and vice-versa
                    load_mask = self.agent_system.unit_agents["agent_0"].get_transport_mask(
                        self.units, game_info, is_transport=is_transport_actor, active_unit=active_unit, cast_range=load_range
                    )
                    final_masks.append(load_mask)
            else:
                # Most other heads are unrestricted spatially
                final_masks.append(unrestricted_mask)
        
        # --- CATEGORICAL HEADS MASKING ---
        categorical_masks = []
        for head in getattr(self, "categorical_heads", []):
            if head == "unload":
                # Valid only if:
                # 1. Active unit is a transport
                # 2. It has cargo
                # 3. It is over safe ground (according to the advanced unload mask)
                if active_unit and active_unit.cargo_max > 0 and active_unit.cargo_used > 0:
                    try:
                        unit_agent = self.agent_system.unit_agents["agent_0"]
                        # 1. Generate the advanced unload mask (Terrain erosion + Building Footprints)
                        # We include ALL potential blockers
                        blockers = self.structures + self.enemy_structures + self.mineral_field + self.vespene_geyser
                        game_info = getattr(self, "game_info", getattr(self, "_game_info", None))
                        
                        full_unload_mask = unit_agent.get_unload_mask(blockers, game_info)
                        
                        # 2. Map current position to grid index
                        grid_idx = unit_agent.get_grid_index_at_world_pos(active_unit.position)
                        
                        # 3. Sample the mask
                        can_unload = bool(full_unload_mask[grid_idx])
                        categorical_masks.append(can_unload)
                        
                        if self.state.game_loop % 100 == 0:
                            logger.debug(f"Advanced Categorical UNLOAD mask for {active_unit.type_id} at index {grid_idx}: {can_unload} (Cargo: {active_unit.cargo_used})")
                    except Exception as e:
                        logger.error(f"Error calculating advanced unload mask: {e}")
                        categorical_masks.append(True)
                else:
                    categorical_masks.append(False)
            else:
                # Default to True for unknown categorical heads
                categorical_masks.append(True)

        return np.concatenate(final_masks + [np.array(categorical_masks)])

    def _get_observation(self) -> Tuple[np.ndarray, Dict]:
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        
        # Flexibly find beacon
        self.beacon = None
        if self.units(UnitTypeId.BEACON_TERRANSMALL).exists:
            self.beacon = self.units(UnitTypeId.BEACON_TERRANSMALL).first
        elif self.structures(UnitTypeId.BEACON_TERRANSMALL).exists:
            self.beacon = self.structures(UnitTypeId.BEACON_TERRANSMALL).first
        elif self.enemy_units(UnitTypeId.BEACON_TERRANSMALL).exists:
            self.beacon = self.enemy_units(UnitTypeId.BEACON_TERRANSMALL).first
        elif self.enemy_structures(UnitTypeId.PYLON).exists:
            self.beacon = self.enemy_structures(UnitTypeId.PYLON).first
        elif self.structures(UnitTypeId.PYLON).exists:
             self.beacon = self.structures(UnitTypeId.PYLON).first

        # Initialize previous_distance on first observation if zergling and beacon are present
        if self.previous_distance is None and self.zergling and self.beacon:
            self.previous_distance = ((self.zergling.position.x - self.beacon.position.x)**2 + (self.zergling.position.y - self.beacon.position.y)**2)**0.5

        enemies = [self.beacon] if self.beacon else []
        
        # Unified Observation: command_target is the GOAL (Beacon), not the last action.
        target_pos = self.beacon.position if self.beacon else self.last_target_pos
        
        raw_obs = {
            'units': self.units,
            'enemy_units': enemies,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos
        }
        
        processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
        return processed_obs, {}

    def _calculate_reward(self) -> Tuple[float, bool]:
        if not self.zergling or not self.beacon:
            return 0, False # No reward if zergling or beacon are missing

        # Load reward constants from training_conf or use defaults
        rewards_conf = self.training_conf.get('rewards', {})
        success_reward = rewards_conf.get('success', 100.0)
        step_penalty = rewards_conf.get('step_penalty', -0.05)
        dist_multiplier = rewards_conf.get('distance_multiplier', 0.1)

        # Manual distance calculation to ensure parity with master branch
        current_dist = ((self.zergling.position.x - self.beacon.position.x) ** 2 + (self.zergling.position.y - self.beacon.position.y) ** 2)**0.5
        reward = 0.0
        reached_target = False

        # Win Condition: Agent reached the beacon
        if current_dist < 2.0:  # Slightly larger radius to ensure hit
            logger.info("SUCCESS! Reached Beacon.")
            reward = success_reward
            reached_target = True
        elif self.previous_distance is not None:
            # Reward for getting closer (positive) or penalty for moving away (negative)
            reward = (self.previous_distance - current_dist) * dist_multiplier # Scale the reward

        # Add constant step penalty to encourage speed
        reward += step_penalty

        # Update previous distance for the next step
        self.previous_distance = current_dist

        return reward, reached_target

    def _get_step_results(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, info = self._get_observation()
        reward, reached_target = self._calculate_reward()
        done = False
        truncated = False
        if reached_target:
            done = True
        return obs, reward, done, truncated, info


class UnitControlBot(BotAI):
    """
    Standalone bot implementation for visualizer and demo modes.
    Directly uses HierarchicalAgentSystem.
    """
    def __init__(self, agent_system: HierarchicalAgentSystem, agent_id: str = "agent_0"):
        super().__init__()
        self.agent_system = agent_system
        self.agent_id = agent_id
        
    async def on_step(self, iteration: int):
        # 1. Capture Raw Observation components
        raw_obs = {
            'units': self.units,
            'enemy_units': self.enemy_units,
            'structures': self.structures,
            'enemy_structures': self.enemy_structures,
            'effects': self.state.effects,
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': None # Optional: Set to current objective if known
        }
        
        # 2. Process with Hierarchical System
        # This also updates the agent's internal state for UI visualization
        results = self.agent_system.process_observation(raw_obs, player_id=1)
        
        # 3. (Optional) Here you would get action from model
        # For now, this bot just allows the UI to visualize what the agent sees

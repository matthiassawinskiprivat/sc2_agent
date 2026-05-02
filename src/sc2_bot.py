"""
A simple bot that acts as a wrapper for the HierarchicalAgentSystem,
and handles the communication with the main process.
"""

import numpy as np
from collections import Counter
from typing import Any, Dict, Optional, Tuple
import multiprocessing
import multiprocessing.connection
import random
from loguru import logger

from sc2.bot_ai import BotAI
from sc2.data import Alliance, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from src.hierarchical_agent import HierarchicalAgentSystem
from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    build_action_head_manifest,
)

# Map props / destructibles that are not Neutral but do not behave like trainable
# units: debug_kill often does not remove them, and they are static geometry.
MAP_CLEANUP_IGNORE_TYPE_IDS = frozenset(
    tid
    for tid in (
        getattr(UnitTypeId, "COLLAPSIBLEROCKTOWER", None),
        getattr(UnitTypeId, "COLLAPSIBLEROCKTOWERDIAGONAL", None),
        getattr(UnitTypeId, "COLLAPSIBLEROCKTOWERDEBRIS", None),
        getattr(UnitTypeId, "COLLAPSIBLEROCKTOWERPUSHUNIT", None),
        getattr(UnitTypeId, "COLLAPSIBLEROCKTOWERPUSHUNITRAMPLEFT", None),
        getattr(UnitTypeId, "COLLAPSIBLEROCKTOWERPUSHUNITRAMPRIGHT", None),
        getattr(UnitTypeId, "COLLAPSIBLETERRANTOWER", None),
        getattr(UnitTypeId, "COLLAPSIBLETERRANTOWERDIAGONAL", None),
        # Purifier tower map props are static doodads on some combat maps and are
        # not scenario units spawned by reset logic.
        getattr(UnitTypeId, "COLLAPSIBLEPURIFIERTOWERDIAGONAL", None),
        getattr(UnitTypeId, "COLLAPSIBLEPURIFIERTOWERVERTICAL", None),
        getattr(UnitTypeId, "COLLAPSIBLEPURIFIERTOWERHORIZONTAL", None),
    )
    if tid is not None
)


def unit_is_subject_to_map_cleanup(u) -> bool:
    """True if reset should try debug_kill on this unit (shared by SC2Bot and Level7Bot)."""
    return u.alliance != Alliance.Neutral and u.type_id not in MAP_CLEANUP_IGNORE_TYPE_IDS


class MapCleanupFailedError(RuntimeError):
    """
    Raised when debug_kill_unit could not clear all non-neutral units within
    the retry budget and strict_empty_map=True. Level0Env treats this like a
    failed reset: close SC2 and relaunch (hard reset).
    """

    def __init__(self, remaining_count: int):
        self.remaining_count = remaining_count
        super().__init__(
            f"map cleanup failed: {remaining_count} killable unit(s) still present after max attempts"
        )


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
        self._game_state_initialized = False
        
        # Load Action Heads — prefer flattened lists from Level0Env (cumulative pipeline
        # heads). Nested env.action_heads is only the *new* heads for that YAML; using it
        # first breaks index alignment with the policy (e.g. level_attack_0 adds attack_local).
        env_conf = self.training_conf.get("env", {})
        self.action_heads = self.training_conf.get("action_heads") or env_conf.get(
            "action_heads", ["move"]
        )
        self.categorical_heads = self.training_conf.get("categorical_heads") or env_conf.get(
            "categorical_heads", []
        )

        if self.training_conf.get("agent_architecture") == "universal":
            manifest = build_action_head_manifest(self.training_conf, list(self.action_heads))
            self.training_conf["action_head_manifest"] = manifest
            self.training_conf.setdefault("fixed_universal_command_dim", FIXED_UNIVERSAL_COMMAND_DIM)

        # enemy_race must be set to avoid AttributeError in BotAIInternal._prepare_step
        # when it checks for Race.Random to update it from observations.
        self._enemy_race = Race.Random
        
        # Satisfaction for BotAI initialization
        self._initialize_variables()
        
        self.unit_command_uses_self_do = False
        self.actions = []
        self.unit_tags_received_action = set()

    async def on_step(self, iteration: int):
        """
        The SC2Bot loop runs inside the StarCraft II simulation process.
        It waits for commands from the Gymnasium environment process via a Pipe (self.conn).
        """
        if iteration == 0:
            # --- INITIALIZATION ---
            # Try to get grid_size from parent env if possible
            # Default to 128, but strictly enforce 32 for level_0
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

            # Setup the specific scenario (e.g., spawn Zergling/Beacon)
            if not self._game_state_initialized:
                await self.reset_game_state()
                self._game_state_initialized = True
            
            # Send initial observation back to the Environment process
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        # --- THE REMOTE CONTROL LOOP ---
        # The bot process blocks here until the Env process sends a command.
        command, data = self.conn.recv()
        
        if command == "step":
            # Command: "Perform this action index and give me the next observation"
            action = data
            
            # 1. Execute: Convert index to movement and tell the unit to go there
            if self.zergling:
                await self.apply_action(self.zergling.tag, action)
            
            # 2. Results: Gather new pixels, calculate reward, and check if reached target
            obs, reward, done, truncated, info = self._get_step_results()
            
            # 3. Respond: Send the results back to the Env (which returns them to model.learn())
            self.conn.send((obs, reward, done, truncated, info))

        elif command == "reset":
            # Command: "Restart the level"
            await self.reset_game_state()
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        elif command == "close":
            # Command: "Shut down SC2"
            await self.client.leave()

    async def apply_action(self, unit_tag, action):
        """
        ACTUATION: Integer Index -> StarCraft II Ability/Command
        This is where the agent's "will" is translated into game physics.
        """
        if not self.agent_system: return
        
        grid_area = self.grid_size * self.grid_size
        total_spatial_indices = len(self.action_heads) * grid_area
        
        # Find the unit in the current game state
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        cmd = None

        if action < total_spatial_indices:
            # --- CASE A: SPATIAL ACTION (Move/Attack/Load at Position) ---
            # 1. Unpack the head (action type) and the pixel index
            head_idx = action // grid_area
            spatial_idx = action % grid_area
            action_name = self.action_heads[head_idx]
            
            # 2. Resolve target: Convert the 'pixel' back to game (world) coordinates
            world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
            target_pos = Point2((world_x, world_y))
            self.last_target_pos = target_pos 
            
            # 3. Issue Command: Tell the unit to perform the ability at that location
            if action_name == "move":
                cmd = unit.move(target_pos)
            elif action_name == "attack":
                cmd = unit.attack(target_pos)
            elif action_name == "load":
                # Specialized logic for loading units into transports
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
            # --- CASE B: CATEGORICAL ACTION (Logic-based, no position) ---
            cat_idx = action - total_spatial_indices
            if cat_idx >= len(self.categorical_heads):
                logger.error(f"Categorical index {cat_idx} out of bounds")
                return
                
            action_name = self.categorical_heads[cat_idx]
            
            if action_name == "unload":
                # E.g., "Empty all passengers here"
                if unit.cargo_used > 0:
                    mapping = {
                        UnitTypeId.MEDIVAC: AbilityId.UNLOADALLAT_MEDIVAC,
                        UnitTypeId.OVERLORD: AbilityId.UNLOADALLAT_OVERLORD,
                        UnitTypeId.WARPPRISM: AbilityId.UNLOADALLAT_WARPPRISM,
                        UnitTypeId.WARPPRISMPHASING: AbilityId.UNLOADALL_WARPPRISM,
                    }
                    ability = mapping.get(unit.type_id, AbilityId.UNLOADALL)
                    
                    if ability in {AbilityId.UNLOADALLAT_MEDIVAC, AbilityId.UNLOADALLAT_OVERLORD, AbilityId.UNLOADALLAT_WARPPRISM}:
                        cmd = unit(ability, unit.position)
                    else:
                        cmd = unit(ability)

        if cmd:
            # Actually send the byte-packet to the StarCraft II engine
            self.do(cmd)

    async def reset_game_state(self, verify: bool = True, strict_empty_map: bool = True):
        """
        Base reset logic: full vision and kill all non-neutral units.

        Some ladder/editor maps leave non-neutral units that do not disappear in
        time (or ignore debug_kill). Level3 multi-unit training sets
        strict_empty_map=False so we still run cleanup but can spawn the scripted
        scenario afterward without killing the worker.
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
        max_cleanup_attempts = 8
        steps_after_kill = 2

        for attempt in range(max_cleanup_attempts):
            # IMPORTANT: We only try to kill units that ARE killable.
            # Neutral units (minerals, etc) cannot be killed and will cause the loop to fail if we include them.
            killable_units = self.all_units.filter(unit_is_subject_to_map_cleanup)
            
            if not killable_units:
                break
                
            logger.info(f"Cleanup attempt {attempt+1}: Killing {killable_units.amount} units.")
            await self.client.debug_kill_unit([u.tag for u in killable_units])
            for _ in range(steps_after_kill):
                await self.client.step()
            await self._refresh_state()
            
            # Re-check killable units
            killable_units = self.all_units.filter(unit_is_subject_to_map_cleanup)
            if not killable_units:
                break
        
        remaining_killable = self.all_units.filter(unit_is_subject_to_map_cleanup)
        if remaining_killable:
            n = remaining_killable.amount
            type_hist = Counter(
                getattr(u.type_id, "name", str(u.type_id)) for u in remaining_killable
            )
            if strict_empty_map:
                logger.error(
                    f"Failed to clear map after {max_cleanup_attempts} cleanup attempts. Remaining: {n}. "
                    f"Types: {dict(type_hist)}. Raising MapCleanupFailedError so the env can hard-reset SC2."
                )
                raise MapCleanupFailedError(n)
            logger.warning(
                f"Map cleanup incomplete ({n} non-neutral unit(s) left); continuing because "
                f"strict_empty_map=False. Types: {dict(type_hist)}"
            )

    def get_active_units(self):
        """Returns active allied units. Default implementation for single-unit bots."""
        if self.zergling:
            return [self.zergling]
        return []

    async def _refresh_state(self):
        """Refreshes the bot's internal state from a new observation."""
        obs = await self.client.observation()
        from sc2.game_state import GameState
        from src.utils.proto_utils import ProtoWrapper
        # Use existing game_info._proto if possible
        proto = getattr(self.game_info, "_proto", None)
        if not proto:
            # Fallback to fetching fresh game info if proto is missing
            game_info = await self.client.get_game_info()
            proto = game_info._proto
            
        self._prepare_step(GameState(obs.observation), ProtoWrapper(proto))

    def action_masks(self, curriculum_progress: float = 1.0) -> np.ndarray:
        if not self.agent_system or "agent_0" not in self.agent_system.unit_agents:
             # Determine fallback grid_size
             gs = 128
             if hasattr(self, "grid_size"):
                 gs = self.grid_size
             elif self.agent_system:
                 gs = self.agent_system.grid_size
             return np.ones(gs * gs, dtype=bool)

        move_mask = self.agent_system.unit_agents["agent_0"].get_action_mask(self.all_units)
        
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

        # --- GUIDANCE CURRICULUM ---
        guidance_mask = np.ones_like(move_mask, dtype=bool)
        # We ONLY apply the mask if curriculum is enabled AND we are actually in the curriculum phase (progress < 1.0)
        if self.training_conf.get("guidance_curriculum", False) and self.beacon and curriculum_progress < 1.0:
            # 1. Calculate expanding radius in grid units
            # We start at 10.0 grid pixels as requested.
            # Max grid distance from a point in a grid_size x grid_size is sqrt(2)*grid_size
            # We use 1.5 multiplier to ensure full coverage even from corners
            max_grid_dist = self.grid_size * 1.5
            grid_radius = 10.0 + (curriculum_progress ** 2.0) * (max_grid_dist - 5.0)
            
            # 2. Map beacon position to grid
            unit_agent = self.agent_system.unit_agents["agent_0"]
            beacon_grid_idx = unit_agent.get_grid_index_at_world_pos(self.beacon.position)
            bx = beacon_grid_idx % self.grid_size
            by = beacon_grid_idx // self.grid_size
            
            # 3. Create circle mask
            Y, X = np.ogrid[:self.grid_size, :self.grid_size]
            dist_sq = (X - bx)**2 + (Y - by)**2
            guidance_mask = (dist_sq <= (grid_radius ** 2)).flatten()
            
            # Ensure the beacon itself is ALWAYS clickable even at start
            guidance_mask[beacon_grid_idx] = True

        # Debug logging for masking
        if hasattr(self, "state") and self.state.game_loop % 100 == 0:
            b_pos = self.beacon.position if self.beacon else "None"
            logger.debug(f"ActionMask check (Loop {self.state.game_loop}): Unit: {active_unit.type_id if active_unit else 'None'}, MoveMask True: {np.sum(move_mask)}, Guidance Radius (Grid): {grid_radius if 'grid_radius' in locals() else 'N/A'}, Beacon World: {b_pos}, Beacon Grid: ({bx if 'bx' in locals() else 'N/A'}, {by if 'by' in locals() else 'N/A'})")

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
                final_masks.append(move_mask & guidance_mask)
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
                        
                        if hasattr(self, "state") and self.state.game_loop % 100 == 0:
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

    def _get_observation(self, active_unit: Optional[Unit] = None) -> Tuple[np.ndarray, Dict]:
        """
        Gathers raw game state and transforms it into the model's pixel observation.
        """
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        
        # 1. Goal Tracking: Flexibly find beacon or target
        # For multi-unit levels (Level 3+), we pick the beacon CLOSEST to the active unit.
        self.beacon = None
        
        # Collect all potential beacons/targets
        potential_beacons = self.all_units.filter(lambda u: u.type_id in {
            UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON
        })
        
        if potential_beacons.exists:
            if active_unit:
                self.beacon = potential_beacons.closest_to(active_unit)
            else:
                self.beacon = potential_beacons.first
        elif active_unit and self.enemy_units.exists:
            # Fallback for combat levels (Level 7+): closest enemy is the objective
            self.beacon = self.enemy_units.closest_to(active_unit)

        # Initialize distance tracking for rewards
        if self.previous_distance is None and self.zergling and self.beacon:
            self.previous_distance = ((self.zergling.position.x - self.beacon.position.x)**2 + (self.zergling.position.y - self.beacon.position.y)**2)**0.5

        # 2. Layer Preparation: Beacons must be in 'enemy_units' layer for training parity
        enemies = list(self.enemy_units)
        if self.beacon and self.beacon not in self.enemy_units:
            enemies.append(self.beacon)
        
        # command_target is the GOAL (Beacon/Target), not the last action.
        target_pos = self.beacon.position if self.beacon else self.last_target_pos
        
        # Filter allied units to NOT include the beacon if it's owned by the bot
        clean_allied_units = self.units
        if self.beacon:
            clean_allied_units = self.units.filter(lambda u: u.tag != self.beacon.tag)

        # 3. Build Raw Observation Dictionary
        raw_obs = {
            'units': clean_allied_units,
            'enemy_units': enemies,
            'structures': self.structures,
            'enemy_structures': self.enemy_structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos,
        }
        
        # 4. Agent Selection (Level 1+): Model sees which specific unit it is commanding
        if active_unit:
            raw_obs['selected_units'] = [active_unit]
        
        # 5. Transform: pixels!
        processed_obs = self.agent_system.process_observation(observation=raw_obs, active_unit=active_unit)["agent_0"]
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


def resolve_tactical_active_unit_for_debug(bot: Any) -> Tuple[Optional[Unit], Optional[int]]:
    """
    Pick the unit/structure that centers the tactical fovea in debug / UI mode.

    Prefer the client's in-game selection (python-sc2: units/structures with
    is_selected). Fall back to the last tag, then the first owned unit.
    """
    if getattr(bot, "units", None) is not None and bot.units.selected.exists:
        u = bot.units.selected.first
        return u, u.tag
    if getattr(bot, "structures", None) is not None and bot.structures.selected.exists:
        u = bot.structures.selected.first
        return u, u.tag

    tag = getattr(bot, "_active_unit_tag", None)
    active: Optional[Unit] = None
    if tag is not None:
        active = bot.units.find_by_tag(tag)
        if active is None and getattr(bot, "structures", None) is not None:
            active = bot.structures.find_by_tag(tag)

    if active is None and getattr(bot, "units", None) is not None and bot.units.exists:
        active = bot.units.first
        tag = active.tag

    return active, tag


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
        beacon = None
        potential_beacons = self.all_units.filter(lambda u: u.type_id in {
            UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON,
            UnitTypeId.BEACON_TERRAN, UnitTypeId.BEACON_PROTOSS, UnitTypeId.BEACON_ZERG
        })
        if potential_beacons.exists:
            # If multiple, find the one closest to our first unit
            if self.units.exists:
                beacon = potential_beacons.closest_to(self.units.first)
            else:
                beacon = potential_beacons.first

        # Ensure beacon is in enemy_units list for rendering on enemy_unit_density
        enemy_units = list(self.enemy_units)
        if beacon and beacon not in self.enemy_units:
            enemy_units.append(beacon)

        raw_obs = {
            'units': self.units,
            'enemy_units': enemy_units,
            'structures': self.structures,
            'enemy_structures': self.enemy_structures,
            'effects': self.state.effects,
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': self.state.creep,
            'command_target': beacon # Set to beacon so it shows on COMMAND_OBJECTIVE
        }

        # 2. Tactical fovea + COMMAND_ASSIGNMENT follow client selection when possible
        if not hasattr(self, "_active_unit_tag"):
            self._active_unit_tag = None

        active_unit, self._active_unit_tag = resolve_tactical_active_unit_for_debug(self)

        selected_units = []
        if self.units.selected.exists:
            selected_units.extend(list(self.units.selected))
        if self.structures.selected.exists:
            selected_units.extend(list(self.structures.selected))
        if not selected_units and active_unit is not None:
            selected_units = [active_unit]
        raw_obs['selected_units'] = selected_units

        results = self.agent_system.process_observation(raw_obs, player_id=1, active_unit=active_unit)
        
        # 3. (Optional) Here you would get action from model
        # For now, this bot just allows the UI to visualize what the agent sees

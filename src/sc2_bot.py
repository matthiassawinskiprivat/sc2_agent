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

        # enemy_race must be set to avoid AttributeError in BotAIInternal._prepare_step
        # when it checks for Race.Random to update it from observations.
        self.enemy_race = Race.Random
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
            grid_size = 128
            if hasattr(self, "_bot_object") and hasattr(self._bot_object, "grid_size"):
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
        Handles dynamic actions based on configured heads.
        """
        if not self.agent_system: return
        
        grid_area = self.grid_size * self.grid_size
        head_idx = action // grid_area
        spatial_idx = action % grid_area
        
        if head_idx >= len(self.action_heads):
            logger.error(f"Action index {head_idx} out of bounds for heads {self.action_heads}")
            return
            
        action_name = self.action_heads[head_idx]
        
        # Resolve target
        world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
        target_pos = Point2((world_x, world_y))
        self.last_target_pos = target_pos 
        
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        cmd = None
        if action_name == "move":
            cmd = unit.move(target_pos)
        elif action_name == "attack":
            cmd = unit.attack(target_pos)
        elif action_name == "load": # Smart on Unit
            candidates = self.units.filter(lambda u: u.distance_to(target_pos) < 3.0)
            transport = next((u for u in candidates if u.cargo_max > 0), None)
            if transport:
                cmd = unit.smart(transport)
            else:
                cmd = unit.move(target_pos)
        elif action_name == "unload":
            if unit.cargo_used > 0:
                mapping = {
                    UnitTypeId.MEDIVAC: AbilityId.UNLOADALLAT_MEDIVAC,
                    UnitTypeId.OVERLORD: AbilityId.UNLOADALLAT_OVERLORD,
                    UnitTypeId.WARPPRISM: AbilityId.UNLOADALLAT_WARPPRISM,
                    UnitTypeId.WARPPRISMPHASING: AbilityId.UNLOADALLAT_WARPPRISM,
                }
                
                ability = mapping.get(unit.type_id, AbilityId.UNLOADALL)
                if ability in {AbilityId.UNLOADALLAT_MEDIVAC, AbilityId.UNLOADALLAT_OVERLORD, AbilityId.UNLOADALLAT_WARPPRISM}:
                    cmd = unit(ability, target_pos)
                else:
                    cmd = unit(ability)
        
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
        
        # AIR UNIT BYPASS: If unit is flying, it can move anywhere (ignore pathing)
        active_units = self.get_active_units()
        idx = getattr(self, "current_unit_index", 0)
        if 0 <= idx < len(active_units):
            if active_units[idx].is_flying:
                move_mask = np.ones_like(move_mask, dtype=bool)

        unrestricted_mask = np.ones_like(move_mask, dtype=bool)
        
        final_masks = []
        for head in self.action_heads:
            if head == "move":
                final_masks.append(move_mask)
            else:
                # Most other heads are unrestricted spatially
                final_masks.append(unrestricted_mask)
                
        return np.concatenate(final_masks)

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

"""
StarCraft 2 Universal Bot
Specialized for the Command + Shared Spatial architecture.
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


class SC2BotUniversal(BotAI):
    """A bot specialized for the Universal agent (MultiDiscrete action space)."""

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
        
        # Load Action Heads (Robustly handle flattened or nested config)
        self.action_heads = self.training_conf.get("action_heads") or \
                           self.training_conf.get("env", {}).get("action_heads", ["move_global"])
        
        self.categorical_heads = self.training_conf.get("categorical_heads") or \
                                self.training_conf.get("env", {}).get("categorical_heads", [])
        
        self.spatial_res = 32

        self._enemy_race = Race.Random
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
            # INITIALIZATION
            self.agent_system = HierarchicalAgentSystem(
                map_size=(128, 128),
                grid_size=32, # Matches Universal Agent Action Space
                training_conf=self.training_conf
            )
            self.agent_system.create_unit_agent("agent_0")
            
            if not self._game_state_initialized:
                await self.reset_game_state()
                self._game_state_initialized = True
            
            obs, info = self._get_observation()
            self.conn.send((obs, info))

        # REMOTE CONTROL LOOP
        command, data = self.conn.recv()
        
        if command == "step":
            if self.zergling:
                await self.apply_action(self.zergling.tag, data)
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
        Unpacks Universal action [command_idx, spatial_idx]
        """
        if not self.agent_system: return
        agent = self.agent_system.unit_agents["agent_0"]
        unit = self.units.find_by_tag(unit_tag)
        if not unit: return

        # 1. Unpack
        head_idx = int(action[0])
        spatial_idx = int(action[1])
        
        # Robust attribute access for delegation
        action_heads = getattr(self, "action_heads", [])
        categorical_heads = getattr(self, "categorical_heads", [])

        if head_idx < len(action_heads):
            head_name = action_heads[head_idx]
            # 2. Convert 16x16 to world
            world_x, world_y = agent.convert_spatial_action_to_world(head_name, spatial_idx, active_unit=unit)
            target_pos = Point2((world_x, world_y))
            self.last_target_pos = target_pos

            # 3. Issue Command
            if "move" in head_name:
                self.do(unit.move(target_pos))
            elif "attack" in head_name:
                self.do(unit.attack(target_pos))
            elif "unload" in head_name:
                # SPATIAL UNLOAD
                mapping = {
                    UnitTypeId.MEDIVAC: AbilityId.UNLOADALLAT_MEDIVAC,
                    UnitTypeId.OVERLORD: AbilityId.UNLOADALLAT_OVERLORD,
                    UnitTypeId.WARPPRISM: AbilityId.UNLOADALLAT_WARPPRISM,
                    UnitTypeId.WARPPRISMPHASING: AbilityId.UNLOADALL_WARPPRISM,
                }
                ability = mapping.get(unit.type_id, AbilityId.UNLOADALLAT)
                self.do(unit(ability, target_pos))
            elif "load" in head_name:
                # Basic load logic
                candidates = self.units.filter(lambda u: u.distance_to(target_pos) < 3.0)
                target_transport = next((u for u in candidates if u.cargo_max > 0), None)
                if target_transport: self.do(unit.smart(target_transport))
                else: self.do(unit.move(target_pos))
        else:
            # Categorical
            cat_idx = head_idx - len(action_heads)
            if cat_idx < len(categorical_heads):
                cat_name = categorical_heads[cat_idx]
                if cat_name == "unload":
                    self.do(unit(AbilityId.UNLOADALL, unit.position))

    async def reset_game_state(self, verify: bool = True):
        self.previous_distance = None
        
        # Clear local unit lists
        from sc2.units import Units
        self.units = Units([], self)
        self.structures = Units([], self)
        self.enemy_units = Units([], self)
        self.enemy_structures = Units([], self)
        self.all_units = Units([], self)

        await self.client.debug_show_map()
        await self.client.step()
        await self._refresh_state()
        
        # Robust Cleanup
        max_cleanup_attempts = 5
        from sc2.data import Alliance
        for attempt in range(max_cleanup_attempts):
            killable_units = self.all_units.filter(lambda u: u.alliance != Alliance.Neutral)
            if not killable_units:
                break
                
            logger.info(f"Universal Cleanup attempt {attempt+1}: Killing {killable_units.amount} units.")
            await self.client.debug_kill_unit([u.tag for u in killable_units])
            await self.client.step()
            await self._refresh_state()
            
            killable_units = self.all_units.filter(lambda u: u.alliance != Alliance.Neutral)
            if not killable_units:
                break

    async def _refresh_state(self):
        obs = await self.client.observation()
        from sc2.game_state import GameState
        from src.utils.proto_utils import ProtoWrapper
        proto = getattr(self.game_info, "_proto", None)
        if not proto:
            game_info = await self.client.get_game_info()
            proto = game_info._proto
        self._prepare_step(GameState(obs.observation), ProtoWrapper(proto))

    def action_masks(self, curriculum_progress: float = 1.0) -> np.ndarray:
        """
        Universal MultiDiscrete masking requires a FLAT concatenated array for 
        vectorized environment compatibility (np.stack).
        Space: [N_Commands, Spatial^2] -> Mask Size: N_Commands + Spatial^2
        """
        # Robust resolution discovery for delegation
        action_heads = getattr(self, "action_heads", [])
        spatial_res = 32

        n_cmds = len(action_heads)
        n_spatial = spatial_res * spatial_res

        # Return a single concatenated array
        return np.concatenate([
            np.ones(n_cmds, dtype=bool),
            np.ones(n_spatial, dtype=bool)
        ])

    def _get_observation(self, active_unit: Optional[Unit] = None) -> Tuple[np.ndarray, Dict]:
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        self.beacon = self.all_units.filter(lambda u: u.type_id in {UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON}).first if self.all_units.filter(lambda u: u.type_id in {UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON}).exists else None
        
        if self.previous_distance is None and self.zergling and self.beacon:
            self.previous_distance = self.zergling.distance_to(self.beacon)

        target_pos = self.beacon.position if self.beacon else self.last_target_pos
        enemies = list(self.enemy_units)
        if self.beacon and self.beacon not in self.enemy_units: enemies.append(self.beacon)

        raw_obs = {
            'units': self.units.filter(lambda u: "BEACON" not in getattr(u.type_id, "name", "").upper() and u.type_id != UnitTypeId.PYLON),
            'enemy_units': enemies,
            'structures': self.structures,
            'enemy_structures': self.enemy_structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos,
            'high_level_command': self.training_conf.get('high_level_command', 0),
            'selected_units': [active_unit] if active_unit else ([self.zergling] if self.zergling else [])
        }
        # Default active unit for tactical view
        if active_unit is None:
            active_unit = self.zergling
            
        processed_obs = self.agent_system.process_observation(observation=raw_obs, active_unit=active_unit)["agent_0"]
        return processed_obs, {}

    def _calculate_reward(self) -> Tuple[float, bool]:
        if not self.zergling or not self.beacon: return 0, False
        rewards_conf = self.training_conf.get('rewards', {})
        current_dist = self.zergling.distance_to(self.beacon)
        reward = rewards_conf.get('step_penalty', -0.05)
        reached_target = False
        if current_dist < 2.0:
            reward = rewards_conf.get('success', 100.0)
            reached_target = True
        elif self.previous_distance is not None:
            reward += (self.previous_distance - current_dist) * rewards_conf.get('distance_multiplier', 0.1)
        self.previous_distance = current_dist
        return reward, reached_target

    def _get_step_results(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, info = self._get_observation()
        reward, reached_target = self._calculate_reward()
        return obs, reward, reached_target, False, info

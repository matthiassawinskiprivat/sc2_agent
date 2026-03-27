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
        # `python-sc2`'s Client.debug_show_map() toggles (2nd call disables again).
        # Track it so repeated env resets do not flip visibility off.
        self._debug_show_map_enabled = False
        
        # Load Action Heads (Robustly handle flattened or nested config)
        self.action_heads = self.training_conf.get("action_heads") or \
                           self.training_conf.get("env", {}).get("action_heads", ["move_global"])
        
        self.categorical_heads = self.training_conf.get("categorical_heads") or \
                                self.training_conf.get("env", {}).get("categorical_heads", [])
        
        self.spatial_res = 32

        # `python-sc2` changed internal initialization across versions.
        # In your current environment, `BotAI` does not expose `_initialize_variables`,
        # and calling it crashes before the game starts.
        self.distance_calculation_method = 2
        self.unit_command_uses_self_do = False
        if hasattr(self, "_initialize_variables"):
            self._initialize_variables()

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
        # Ensure full visibility: debug_show_map toggles, so call only once per game.
        if not self._debug_show_map_enabled:
            await self.client.debug_show_map()
            self._debug_show_map_enabled = True
        await self.client.step()
        await self._refresh_state()

        # Deterministic spawn:
        # 1) Kill any existing allied zergling/beacon units that may have been
        #    pre-placed on the map (or from earlier resets).
        # 2) Spawn exactly ONE zergling and ONE beacon on pathable terrain.
        area = getattr(self.game_info, "playable_area", None)
        grid = getattr(self.game_info, "pathing_grid", None)
        player_id = getattr(self, "player_id", 1)

        if area is not None and grid is not None:
            margin = 5.0

            # Cache valid spawn points — pathing grid is static during training.
            # Re-scanning the whole grid on every episode reset is the primary
            # cause of FPS degradation as episodes get shorter with a better agent.
            if not hasattr(self, "_cached_spawn_points") or not self._cached_spawn_points:
                pts = []
                for x in range(int(area.x + margin), int(area.x + area.width - margin), 2):
                    for y in range(int(area.y + margin), int(area.y + area.height - margin), 2):
                        try:
                            if grid[(x, y)] != 0:
                                pts.append(Point2((x, y)))
                        except Exception:
                            continue
                self._cached_spawn_points = pts
            valid_points = self._cached_spawn_points

            def _rand_pathable_point(max_tries: int = 50) -> Point2:
                for _ in range(max_tries):
                    x = random.uniform(float(area.x + margin), float(area.x + area.width - margin))
                    y = random.uniform(float(area.y + margin), float(area.y + area.height - margin))
                    try:
                        if grid[(int(x), int(y))] != 0:
                            return Point2((x, y))
                    except Exception:
                        pass
                # Hard fallback: center of playable area
                return Point2((float(area.x + area.width / 2), float(area.y + area.height / 2)))

            if len(valid_points) >= 2:
                p1, p2 = random.sample(valid_points, 2)
                max_attempts = 25
                while p1.distance_to(p2) < 10.0 and max_attempts > 0:
                    p1, p2 = random.sample(valid_points, 2)
                    max_attempts -= 1
            elif len(valid_points) == 1:
                p1 = valid_points[0]
                p2 = _rand_pathable_point()
            else:
                p1 = _rand_pathable_point()
                p2 = _rand_pathable_point()
        else:
            # Worst-case fallback if geometry is missing.
            p1 = Point2((0, 0))
            p2 = Point2((0, 0))

        # Kill pre-existing allied controlled units/beacons.
        kill_tags = []
        if self.units(UnitTypeId.ZERGLING).exists:
            kill_tags.extend([u.tag for u in self.units(UnitTypeId.ZERGLING)])
        if self.units(UnitTypeId.BEACON_TERRANSMALL).exists:
            kill_tags.extend([u.tag for u in self.units(UnitTypeId.BEACON_TERRANSMALL)])
        if self.units(UnitTypeId.PYLON).exists:
            kill_tags.extend([u.tag for u in self.units(UnitTypeId.PYLON)])

        if kill_tags:
            await self.client.debug_kill_unit(kill_tags)
            await self.client.step()
            await self._refresh_state()

        logger.info(f"Universal spawn: placing exactly one Zergling + Beacon on pathable tiles: zergling={p1} beacon={p2}")
        await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, p1, player_id]])
        await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, p2, player_id]])
        await self.client.step()
        await self._refresh_state()

        # Update cached references.
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        self.beacon = (
            self.units(UnitTypeId.BEACON_TERRANSMALL).first
            if self.units(UnitTypeId.BEACON_TERRANSMALL).exists
            else (self.units(UnitTypeId.PYLON).first if self.units(UnitTypeId.PYLON).exists else None)
        )

        # Cleanup:
        # For movement levels, the zergling (agent) and the beacon (objective)
        # are typically map-placed. Do NOT delete them, or the env can't recover.
        #
        # Remove other enemy units/structures, except beacon-related ones.
        beacon_type_ids = {UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON}

        # python-sc2 internal attributes differ across versions:
        # - `GameState` may not expose `enemy_units`
        # - bot may or may not have `enemy_units` populated at this point
        enemy_units_obj = getattr(self, "enemy_units", None)
        if enemy_units_obj is None and hasattr(self, "state"):
            enemy_units_obj = getattr(self.state, "enemy_units", None)

        # If we can't access enemy unit lists, skip cleanup (better to preserve
        # visibility than to crash or delete critical units).
        if enemy_units_obj is None:
            logger.warning("Universal Cleanup: enemy_units not available; skipping cleanup to preserve visibility.")
            return

        killable_enemy = enemy_units_obj.filter(lambda u: u.type_id not in beacon_type_ids)
        if killable_enemy:
            logger.info(f"Universal Cleanup: Killing {killable_enemy.amount} enemy units (except beacon).")
            await self.client.debug_kill_unit([u.tag for u in killable_enemy])
            await self.client.step()
            await self._refresh_state()

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
        if not self.agent_system or "agent_0" not in self.agent_system.unit_agents:
             # Fallback
             action_heads = getattr(self, "action_heads", [])
             n_cmds = len(action_heads)
             n_spatial = 32 * 32
             return np.concatenate([np.ones(n_cmds, dtype=bool), np.ones(n_spatial, dtype=bool)])

        # 1. Base pathing mask from agent system (32x32)
        # Universal architecture handles spatial masking separately.
        # Passing None ensures we get exactly the 32x32 spatial mask (size 1024).
        move_mask = self.agent_system.unit_agents["agent_0"].get_action_mask(None)
        
        # AIR UNIT BYPASS: Flyers can move anywhere (ignore terrain)
        active_unit = self.zergling # Default for this level
        if active_unit and active_unit.is_flying:
            move_mask = np.ones_like(move_mask, dtype=bool)

        # 2. Guidance Curriculum (Circle mask around beacon)
        guidance_mask = np.ones_like(move_mask, dtype=bool)
        if self.training_conf.get("guidance_curriculum", False) and self.beacon and curriculum_progress < 1.0:
            max_grid_dist = 32 * 1.5
            grid_radius = 10.0 + (curriculum_progress ** 2.0) * (max_grid_dist - 10.0)
            
            unit_agent = self.agent_system.unit_agents["agent_0"]
            beacon_grid_idx = unit_agent.get_grid_index_at_world_pos(self.beacon.position)
            bx, by = beacon_grid_idx % 32, beacon_grid_idx // 32
            
            Y, X = np.ogrid[:32, :32]
            dist_sq = (X - bx)**2 + (Y - by)**2
            guidance_mask = (dist_sq <= (grid_radius ** 2)).flatten()
            guidance_mask[beacon_grid_idx] = True

        # Combine Move and Guidance
        final_spatial_mask = move_mask & guidance_mask

        # 3. Command Head Masking (Categorical)
        # Universal architecture only uses action_heads for its MultiDiscrete space.
        # Categorical heads (like unload) are NOT part of the space definition in Level0Env.
        action_heads = self.training_conf.get('action_heads', [])
        
        if not action_heads:
            # Fallback to defaults if not in training_conf
            action_heads = getattr(self, "action_heads", ["move_global"])
        
        # Ensure we return a mask of length len(action_heads) + final_spatial_mask.size
        # This MUST match sum(action_space.nvec) = len(action_heads) + 1024
        cmd_mask = np.ones(len(action_heads), dtype=bool)
        
        # Return a single concatenated array [Cmds, Spatial]
        mask_out = np.concatenate([cmd_mask, final_spatial_mask])
        if len(mask_out) != 1026:
             logger.warning(f"CRITICAL SHAPE: cmd_mask={len(cmd_mask)}, spatial={final_spatial_mask.shape}, action_heads={action_heads}")
        return mask_out

    def _get_observation(self, active_unit: Optional[Unit] = None) -> Tuple[np.ndarray, Dict]:
        self.zergling = self.units(UnitTypeId.ZERGLING).first if self.units(UnitTypeId.ZERGLING).exists else None
        self.beacon = self.units.filter(lambda u: u.type_id in {UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON}).first if self.units.filter(lambda u: u.type_id in {UnitTypeId.BEACON_TERRANSMALL, UnitTypeId.PYLON}).exists else None
        
        if self.previous_distance is None and self.zergling and self.beacon:
            self.previous_distance = self.zergling.distance_to(self.beacon)

        target_pos = self.beacon.position if self.beacon else self.last_target_pos
        # python-sc2 internal attributes differ across versions.
        enemy_units_obj = getattr(self, "enemy_units", None)
        if enemy_units_obj is None and hasattr(self, "state"):
            enemy_units_obj = getattr(self.state, "enemy_units", None)

        enemies = list(enemy_units_obj) if enemy_units_obj is not None else []
        if self.beacon and (enemy_units_obj is None or self.beacon not in enemy_units_obj):
            enemies.append(self.beacon)

        enemy_structures_obj = getattr(self, "enemy_structures", None)
        if enemy_structures_obj is None:
            enemy_structures_obj = (
                enemy_units_obj.structure
                if enemy_units_obj is not None and hasattr(enemy_units_obj, "structure")
                else []
            )
        
        game_data_obj = getattr(self, "game_data", None)
        if game_data_obj is None:
            game_data_obj = getattr(self, "_game_data", None)

        raw_obs = {
            'units': self.units.filter(lambda u: "BEACON" not in getattr(u.type_id, "name", "").upper() and u.type_id != UnitTypeId.PYLON),
            'enemy_units': enemies,
            'structures': self.units.structure,
            'enemy_structures': enemy_structures_obj,
            'effects': getattr(self.state, "effects", []),
            'game_info': self.game_info,
            'game_data': game_data_obj,
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
        rewards_conf = self.training_conf.get('rewards', {})
        reward = rewards_conf.get('step_penalty', -0.05)
        reached_target = False
        
        if not self.zergling or not self.beacon: 
             # Map trigger deletion check
             if self.previous_distance is not None and self.previous_distance < 4.0:
                 logger.info(f"SUCCESS (Inferred)! Reached Beacon (prev dist: {self.previous_distance:.2f})")
                 reward = rewards_conf.get('success', 100.0)
                 return reward, True
             return 0, False
             
        current_dist = self.zergling.distance_to(self.beacon)
        if current_dist < 2.0:
            logger.info(f"SUCCESS! Reached Beacon {self.beacon.tag} (Distance: {current_dist:.2f})")
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

"""Zerg build-agent bot: spawn, observations, actuation, rewards."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.sc2_bot import MapCleanupFailedError, unit_is_subject_to_map_cleanup
from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    BUILD_PLACEMENT_CELLS,
    MAX_BUILD_WORKERS,
)
from src.agent.commands import Command, CommandType
from src.runtime.command_queue import CommandQueue
from src.runtime.mining_controller import MiningController, NoOpMiningController
from src.runtime.scripted_manager import ScriptedManager
from src.runtime.squad_registry import SquadRegistry
from src.runtime.worker_registry import WorkerRegistry, WorkerRole
from src.transformers.build_observation_builder import BuildObservationBuilder

# SC2 debug_create_unit owner for neutral map resources (geysers / minerals).
_NEUTRAL_RESOURCE_PLAYER = 3


class ZergBuildBot(BotAI):
    """Build curriculum bot with scripted manager and agent-faithful build actuation."""

    def __init__(self, conn=None, training_conf: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.conn = conn
        self.training_conf = training_conf or {}
        # Gym env drives the bot directly (not run_game); initialize BotAI unit caches.
        self._enemy_race = Race.Random
        self._initialize_variables()
        self.unit_command_uses_self_do = False
        self.actions: list = []
        self.worker_registry = WorkerRegistry()
        self.squad_registry = SquadRegistry()
        self.command_queue = CommandQueue()
        env_cfg = self._level_env_config()
        if env_cfg.get("manager_script"):
            self.scripted_manager = ScriptedManager.from_level_config(env_cfg)
            self.scripted_manager.worker_registry = self.worker_registry
            self.scripted_manager.command_queue = self.command_queue
        else:
            self.scripted_manager = ScriptedManager(
                script_steps=[],
                macro_tick_interval=int(env_cfg.get("macro_tick_interval", 8)),
                worker_registry=self.worker_registry,
                command_queue=self.command_queue,
            )
        self.obs_builder = BuildObservationBuilder(self.training_conf)
        self.mining_controller: MiningController = self._resolve_mining_controller()
        self.rewards_cfg = self.training_conf.get("rewards", {})
        self.episode_step = 0
        self._last_build_attempt_failed = False
        self._structures_at_step_start = 0
        self._focus_worker_tag = -1
        self._spawn_hatchery_pos: Optional[Point2] = None
        self._target_geyser_pos: Optional[Point2] = None
        self._game_state_initialized = False
        self._cached_spawn_geometry_key: Optional[str] = None
        self._cached_hatch_spawn_pos: Optional[Point2] = None
        self._debug_show_map_enabled = False
        self._debug_spawned_neutral_tags: set[int] = set()

    async def on_step(self, iteration: int):
        """Gym env calls on_step(0) once after _prepare_start to spawn the scenario."""
        if iteration == 0:
            if not self._game_state_initialized:
                # Fresh create_game session: map is empty; skip debug_kill loop.
                await self.reset_game_state(skip_cleanup=True)
                self._game_state_initialized = True
            return

    _LEVEL_ENV_KEYS = (
        "manager_script",
        "macro_tick_interval",
        "spawn",
        "pre_spawn_structures",
        "enemy_rush",
        "mining_controller",
        "spawn_resources_if_missing",
    )

    def _level_env_config(self) -> Dict[str, Any]:
        """Merge nested env dict with flattened root keys (BuildEnv / Level0Env contract)."""
        cfg: Dict[str, Any] = {}
        nested = self.training_conf.get("env", {})
        if isinstance(nested, dict):
            cfg.update(nested)
        for key in self._LEVEL_ENV_KEYS:
            if key in self.training_conf:
                cfg[key] = self.training_conf[key]
        return cfg

    def _spawn_config(self) -> Dict[str, Any]:
        """Read spawn block from flattened training_conf or nested env.spawn."""
        env_cfg = self._level_env_config()
        spawn = env_cfg.get("spawn")
        return spawn if isinstance(spawn, dict) else {}

    def _mining_controller_name(self) -> str:
        name = self._level_env_config().get("mining_controller")
        return str(name) if name else "noop"

    def _resolve_mining_controller(self) -> MiningController:
        controller_name = self._mining_controller_name()
        if controller_name == "scripted":
            from src.runtime.mining_controller import ScriptedMiningController

            return ScriptedMiningController()
        return NoOpMiningController()

    async def reset_game_state(self, verify: bool = True, skip_cleanup: bool = False):
        await self._ensure_map_visible()
        if not skip_cleanup:
            await self._clear_debug_spawned_neutrals()
            await self._clear_map_units()
            await self._ensure_own_workers_cleared()
        self._clear_spawn_session_state()
        self.worker_registry.clear()
        self.command_queue.clear()
        self.scripted_manager.reset()
        self.episode_step = 0
        self._last_build_attempt_failed = False
        await self._spawn_scenario()
        self._structures_at_step_start = len(self.structures)
        self._patch_script_targets()
        self._assign_initial_worker_roles()
        self.scripted_manager._emit_step_if_available()
        active = self.command_queue.active
        if active and active.unit_type_id is not None:
            level = self.training_conf.get("level_name", "?")
            logger.info(
                f"Build level '{level}' macro command: "
                f"{active.command_type.name} {active.unit_type_id.name} "
                f"target={active.target_pos}"
            )
        await self._kick_economy_orders()

    async def _ensure_map_visible(self) -> None:
        """Reveal the map once per session so neutral minerals/geysers populate bot state."""
        if self._debug_show_map_enabled:
            return
        await self.client.debug_show_map()
        self._debug_show_map_enabled = True
        await self.client.step()
        await self._refresh_state()

    async def _kick_economy_orders(self) -> None:
        """Issue first mining/build orders immediately after spawn so workers do not idle."""
        await self.on_build_substep()
        actions = getattr(self, "actions", None)
        if actions:
            await self.client.actions(actions[:])
            actions.clear()
            await self.client.step()
            await self._refresh_state()

    def _clear_spawn_session_state(self) -> None:
        """Drop per-episode spawn anchors so the next reset re-anchors to the hatch."""
        self._cached_spawn_geometry_key = None
        self._cached_hatch_spawn_pos = None
        self._target_geyser_pos = None
        self._spawn_hatchery_pos = None

    def _tag_debug_spawned_neutrals(self, hatch_pos: Point2, radius: float = 18.0) -> None:
        """Remember debug-spawned minerals/geysers so the next reset can remove them."""
        from sc2.data import Alliance

        for collection in (self.mineral_field, self.vespene_geyser):
            for unit in collection:
                if getattr(unit, "alliance", None) != Alliance.Neutral:
                    continue
                if unit.distance_to(hatch_pos) <= radius:
                    self._debug_spawned_neutral_tags.add(int(unit.tag))

    async def _clear_debug_spawned_neutrals(self) -> None:
        if not self._debug_spawned_neutral_tags:
            return
        await self._refresh_state()
        tags_to_kill = [
            int(u.tag)
            for u in self.all_units
            if int(u.tag) in self._debug_spawned_neutral_tags
        ]
        if tags_to_kill:
            await self.client.debug_kill_unit(tags_to_kill)
            for _ in range(2):
                await self.client.step()
            await self._refresh_state()
        self._debug_spawned_neutral_tags.clear()

    async def _ensure_own_workers_cleared(self) -> None:
        """Kill any surviving drones so respawn count is not short-circuited."""
        await self._refresh_state()
        drones = list(self.workers)
        if not drones:
            return
        await self.client.debug_kill_unit([int(u.tag) for u in drones])
        for _ in range(2):
            await self.client.step()
        await self._refresh_state()

    def _patch_script_targets(self):
        """Replace placeholder YAML targets with spawn-time world positions."""
        hatch = (
            self.townhalls.first.position
            if self.townhalls.exists
            else (self._spawn_hatchery_pos or self.start_location)
        )
        for step in self.scripted_manager.script_steps:
            cmd = step.command
            if cmd.target_pos is None:
                continue
            if cmd.target_pos.x != 0 or cmd.target_pos.y != 0:
                continue
            if cmd.unit_type_id == UnitTypeId.EXTRACTOR and self._target_geyser_pos:
                cmd.target_pos = self._target_geyser_pos
            elif cmd.unit_type_id in {
                UnitTypeId.SPAWNINGPOOL,
                UnitTypeId.HATCHERY,
                UnitTypeId.LAIR,
                UnitTypeId.HIVE,
            }:
                cmd.target_pos = hatch

    def _unit_is_build_scenario_cleanup(self, u) -> bool:
        """Only remove units owned by the training bot (keep map/neutral/other slots)."""
        if not unit_is_subject_to_map_cleanup(u):
            return False
        owner = getattr(u, "owner_id", None)
        if owner is not None:
            return int(owner) == int(self.player_id)
        alliance = getattr(u, "alliance", None)
        from sc2.data import Alliance

        return alliance == Alliance.Self

    async def _clear_map_units(self, strict_empty_map: bool = False):
        """Kill training-bot units so resets do not stack hatcheries."""
        await self._refresh_state()

        max_cleanup_attempts = 8
        steps_after_kill = 2
        for attempt in range(max_cleanup_attempts):
            killable_units = self.all_units.filter(self._unit_is_build_scenario_cleanup)
            if not killable_units:
                break
            alliance_hist = Counter(
                getattr(getattr(u, "alliance", None), "name", str(getattr(u, "alliance", "?")))
                for u in killable_units
            )
            logger.info(
                f"Build cleanup attempt {attempt + 1}: killing {killable_units.amount} units "
                f"(alliance={dict(alliance_hist)})."
            )
            await self.client.debug_kill_unit([u.tag for u in killable_units])
            for _ in range(steps_after_kill):
                await self.client.step()
            await self._refresh_state()

        remaining = self.all_units.filter(self._unit_is_build_scenario_cleanup)
        if remaining:
            n = remaining.amount
            type_hist = Counter(
                getattr(u.type_id, "name", str(u.type_id)) for u in remaining
            )
            if strict_empty_map:
                raise MapCleanupFailedError(n)
            logger.warning(
                f"Build map cleanup incomplete ({n} unit(s) left). Types: {dict(type_hist)}"
            )

    @staticmethod
    def _point_in_playable_area(area, pos: Point2) -> bool:
        if area is None:
            return True
        if hasattr(area, "contains"):
            return area.contains(pos)
        return (
            float(area.x) <= pos.x <= float(area.x) + float(area.width)
            and float(area.y) <= pos.y <= float(area.y) + float(area.height)
        )

    def _grid_pathable(self, x: int, y: int) -> bool:
        grid = self.game_info.pathing_grid
        data = getattr(grid, "data_numpy", None)
        shape = getattr(data, "shape", ())
        if data is not None and len(shape) == 2:
            h, w = shape
            if not (0 <= x < w and 0 <= y < h):
                return False
        try:
            return bool(grid[(x, y)])
        except (IndexError, KeyError):
            return False

    def _is_pathable(self, pos: Point2) -> bool:
        return self._grid_pathable(int(round(pos.x)), int(round(pos.y)))

    def _find_buildable_near(
        self,
        center: Point2,
        min_dist: float = 6.0,
        max_dist: float = 14.0,
    ) -> Point2:
        """Pick a pathable tile near ``center`` for hatchery placement."""
        area = self.game_info.playable_area
        best: Optional[Point2] = None
        best_dist = float("inf")
        for dist in range(int(min_dist), int(max_dist) + 1):
            for angle_deg in range(0, 360, 20):
                rad = math.radians(angle_deg)
                x = int(round(center.x + dist * math.cos(rad)))
                y = int(round(center.y + dist * math.sin(rad)))
                pos = Point2((float(x), float(y)))
                if not self._point_in_playable_area(area, pos):
                    continue
                if not self._grid_pathable(x, y):
                    continue
                d = center.distance_to(pos)
                if min_dist <= d <= max_dist and d < best_dist:
                    best_dist = d
                    best = pos
        if best is not None:
            return best
        for dist in range(int(max_dist) + 1, int(max_dist) + 20):
            for angle_deg in range(0, 360, 15):
                rad = math.radians(angle_deg)
                x = int(round(center.x + dist * math.cos(rad)))
                y = int(round(center.y + dist * math.sin(rad)))
                pos = Point2((float(x), float(y)))
                if not self._point_in_playable_area(area, pos):
                    continue
                if not self._grid_pathable(x, y):
                    continue
                return pos
        return self._default_hatch_location()

    def _spawn_reference_point(self) -> Point2:
        """Anchor for geyser/hatch placement when ``start_location`` is not set yet."""
        start = getattr(self, "start_location", None)
        if start is not None:
            return start
        if self.game_info is not None:
            center = getattr(self.game_info, "map_center", None)
            if center is not None:
                return center
            area = self.game_info.playable_area
            return Point2(
                (float(area.x + area.width / 2), float(area.y + area.height / 2))
            )
        return Point2((64.0, 64.0))

    def _default_hatch_location(self) -> Point2:
        area = self.game_info.playable_area
        for x in range(int(area.x + 4), int(area.x + area.width - 4), 3):
            for y in range(int(area.y + 4), int(area.y + area.height - 4), 3):
                if self._grid_pathable(x, y):
                    return Point2((float(x), float(y)))
        return self._spawn_reference_point()

    def _spawn_geometry_key(self) -> str:
        map_name = getattr(self.game_info, "map_name", "unknown")
        if self._target_geyser_pos is not None:
            gx = int(round(self._target_geyser_pos.x))
            gy = int(round(self._target_geyser_pos.y))
            return f"{map_name}:{gx}:{gy}"
        return str(map_name)

    def _resolve_hatch_spawn_pos(self) -> Point2:
        """Stable hatch location for the session (reuse after first successful spawn)."""
        geom = self._spawn_geometry_key()
        if (
            self._cached_hatch_spawn_pos is not None
            and self._cached_spawn_geometry_key == geom
        ):
            return self._cached_hatch_spawn_pos

        ref = self._spawn_reference_point()
        if self.vespene_geyser.exists:
            geyser = self.vespene_geyser.closest_to(ref)
            self._target_geyser_pos = geyser.position
            hatch_pos = self._find_buildable_near(geyser.position)
        else:
            self._target_geyser_pos = None
            hatch_pos = self._default_hatch_location()

        if not self._is_pathable(hatch_pos):
            hatch_pos = self._default_hatch_location()

        geom = self._spawn_geometry_key()
        self._cached_spawn_geometry_key = geom
        self._cached_hatch_spawn_pos = hatch_pos
        return hatch_pos

    async def _spawn_drones_near_hatch(
        self, hatch_pos: Point2, count: int, player: int
    ) -> None:
        """Spawn drones in a ring around the hatchery with retries if SC2 drops units."""
        max_attempts = 3
        for attempt in range(max_attempts):
            have = len(list(self.workers))
            missing = count - have
            if missing <= 0:
                return

            cmds: List[list] = []
            for i in range(missing):
                idx = have + i
                angle = math.radians(45 + (idx + attempt * count) * (360 / max(count, 1)))
                dist = 4.5 + 0.5 * attempt
                pos = Point2(
                    (
                        hatch_pos.x + dist * math.cos(angle),
                        hatch_pos.y + dist * math.sin(angle),
                    )
                )
                if not self._is_pathable(pos):
                    pos = Point2((hatch_pos.x + 4.0 + idx * 0.6, hatch_pos.y + 2.0))
                cmds.append([UnitTypeId.DRONE, 1, pos, player])

            await self.client.debug_create_unit(cmds)
            await self.client.step()
            await self._refresh_state()

        have = len(list(self.workers))
        if have < count:
            logger.warning(
                f"Build spawn: expected {count} drones, got {have} after {max_attempts} attempts."
            )

    async def _grant_spawn_resources(self, minerals: int, vespene: int = 0) -> None:
        """Grant banked resources via burnysc2 debug cheats (no per-amount API)."""
        if minerals <= 0 and vespene <= 0:
            return
        try:
            if minerals > 0:
                await self.client.debug_minerals()
            if vespene > 0:
                await self.client.debug_gas()
        except Exception as exc:
            logger.warning(f"debug resource grant failed: {exc}")
            return
        await self.client.step()
        await self._refresh_state()

    async def _spawn_resource_nodes(self, hatch_pos: Point2) -> None:
        """Spawn neutral minerals + geyser when the map has no vespene (e.g. blank grids)."""
        geyser_pos = Point2((hatch_pos.x + 6.0, hatch_pos.y + 2.0))
        mineral_offsets = [
            Point2((hatch_pos.x + 4.0, hatch_pos.y - 1.0)),
            Point2((hatch_pos.x + 5.0, hatch_pos.y + 0.5)),
            Point2((hatch_pos.x + 4.5, hatch_pos.y + 2.0)),
        ]
        owner = _NEUTRAL_RESOURCE_PLAYER
        await self.client.debug_create_unit(
            [[UnitTypeId.VESPENEGEYSER, 1, geyser_pos, owner]]
        )
        for mpos in mineral_offsets:
            await self.client.debug_create_unit(
                [[UnitTypeId.MINERALFIELD, 1, mpos, owner]]
            )
        self._target_geyser_pos = geyser_pos
        await self.client.step()
        await self._refresh_state()
        self._tag_debug_spawned_neutrals(hatch_pos)

    async def _spawn_scenario(self):
        spawn = self._spawn_config()
        env_cfg = self._level_env_config()
        num_drones = int(spawn.get("drones", 3))
        minerals = int(spawn.get("minerals", 500))
        vespene = int(spawn.get("vespene", 0))
        pre_structures = env_cfg.get("pre_spawn_structures", [])
        spawn_resources_if_missing = bool(
            spawn.get(
                "spawn_resources_if_missing",
                env_cfg.get("spawn_resources_if_missing", True),
            )
        )

        await self._refresh_state()

        hatch_pos = self._resolve_hatch_spawn_pos()
        self._spawn_hatchery_pos = hatch_pos

        player = self.player_id
        await self.client.debug_create_unit([[UnitTypeId.HATCHERY, 1, hatch_pos, player]])
        await self.client.step()
        await self._refresh_state()

        await self._spawn_drones_near_hatch(hatch_pos, num_drones, player)

        if not self.vespene_geyser.exists and spawn_resources_if_missing:
            await self._spawn_resource_nodes(hatch_pos)

        if self.vespene_geyser.exists and self._target_geyser_pos is None:
            self._target_geyser_pos = self.vespene_geyser.closest_to(hatch_pos).position

        for struct_name in pre_structures:
            struct_type = UnitTypeId[struct_name.upper()]
            offset = Point2((hatch_pos.x + 4.0, hatch_pos.y + 2.0))
            await self.client.debug_create_unit([[struct_type, 1, offset, player]])

        if env_cfg.get("enemy_rush"):
            rush_pos = Point2((hatch_pos.x + 18.0, hatch_pos.y + 4.0))
            enemy_player = 2 if player == 1 else 1
            await self.client.debug_create_unit(
                [[UnitTypeId.ZERGLING, 8, rush_pos, enemy_player]]
            )

        await self.client.step()
        await self._refresh_state()

        await self._grant_spawn_resources(minerals, vespene)

    async def _refresh_state(self):
        """Refresh bot caches from the latest observation (no implicit game step)."""
        obs = await self.client.observation()
        from src.utils.sc2_observation_compat import game_state_from_response_observation
        from src.utils.proto_utils import ProtoWrapper

        state = game_state_from_response_observation(obs.observation)
        self._prepare_step(state, ProtoWrapper(self.game_info._proto))

    def _assign_initial_worker_roles(self):
        spawn = self._spawn_config()
        drones_sorted = sorted(self.workers, key=lambda u: u.tag)
        num_drones = len(drones_sorted)
        mining_count = int(spawn.get("mining_drones", 0))

        if mining_count <= 0 and self._mining_controller_name() == "scripted":
            mining_count = min(2, max(0, num_drones - 1))

        if num_drones > 1:
            mining_count = min(mining_count, num_drones - 1)
        else:
            mining_count = 0

        for i, drone in enumerate(drones_sorted):
            role = WorkerRole.MINING if i < mining_count else WorkerRole.BUILD
            self.worker_registry.assign(drone.tag, role)

    def action_masks(self) -> np.ndarray:
        """Mask padded worker slots; placement grid stays fully available."""
        mask = np.ones(MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS, dtype=bool)
        n_workers = len(list(self.workers))
        if n_workers < MAX_BUILD_WORKERS:
            mask[n_workers:MAX_BUILD_WORKERS] = False
        return mask

    async def on_build_substep(self):
        """Per game-step runtime: manager macro tick, mining, command activation."""
        self.scripted_manager.on_game_step()
        self.command_queue.activate_next()
        await self.mining_controller.step(self, self.worker_registry.mining_tags())
        if self.training_conf.get("defense_mode"):
            self._maybe_reassign_workers_to_micro()

    def _maybe_reassign_workers_to_micro(self):
        """Manager-driven worker defense: assign micro role when enemies are near."""
        if not self.enemy_units.exists:
            return
        threat = self.enemy_units.filter(lambda u: u.can_attack or u.is_army)
        if not threat.exists:
            return
        for worker in self.workers:
            nearest = threat.closest_to(worker.position)
            if worker.distance_to(nearest) < 15.0:
                self.worker_registry.set_role(worker.tag, WorkerRole.MICRO)
                target = nearest.position
                self.squad_registry.set_unit_directive(worker.tag, "attack", target)
                self.do(worker.attack(target))

    def _get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        playable = self.game_info.playable_area
        cmd = self.command_queue.active
        obs = self.obs_builder.build(
            self,
            self.worker_registry,
            cmd,
            playable_size=(playable.width, playable.height),
        )
        return obs, {}

    def _get_step_results(self) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        obs, info = self._get_observation()
        reward = self._compute_reward()
        done = self._check_success()
        if done:
            reward += float(self.rewards_cfg.get("success", 40.0))
        return obs, reward, done, False, info

    def _compute_reward(self) -> float:
        r = float(self.rewards_cfg.get("step_penalty", -0.05))
        if self._last_build_attempt_failed:
            r += float(self.rewards_cfg.get("invalid_build_penalty", -1.0))
        cmd = self.command_queue.active
        if cmd and cmd.target_pos is not None:
            shaping = float(self.rewards_cfg.get("distance_multiplier", 0.0))
            if shaping > 0 and self._focus_worker_tag > 0:
                worker = next(
                    (u for u in self.workers if u.tag == self._focus_worker_tag),
                    None,
                )
                if worker:
                    dist = worker.distance_to(cmd.target_pos)
                    r += shaping * max(0.0, 20.0 - dist) * 0.01
        return r

    def _check_success(self) -> bool:
        cmd = self.command_queue.active
        if cmd is None:
            return False
        if cmd.command_type in {CommandType.BUILD, CommandType.EXPAND}:
            if cmd.unit_type_id is None:
                return False
            if self.structures(cmd.unit_type_id).ready.exists:
                self.command_queue.complete_active()
                return True
        if cmd.command_type == CommandType.TRAIN:
            if cmd.unit_type_id and self.units(cmd.unit_type_id).exists:
                self.command_queue.complete_active()
                return True
        return False

    async def apply_build_action(self, action: np.ndarray) -> bool:
        """Decode policy action into SC2 build order. Returns False if no macro command."""
        cmd = self.command_queue.active
        if cmd is None or not cmd.is_macro:
            return False

        worker_idx = int(action[0])
        placement_idx = int(action[1])
        workers_sorted = sorted(self.workers, key=lambda u: u.tag)
        if worker_idx >= len(workers_sorted):
            self._last_build_attempt_failed = True
            return True
        worker = workers_sorted[worker_idx]
        self._focus_worker_tag = int(worker.tag)

        if cmd.command_type == CommandType.TRAIN:
            return await self._apply_train(cmd, worker)

        target = self.decode_placement(placement_idx, cmd.target_pos or worker.position)
        self._last_build_attempt_failed = False

        if cmd.unit_type_id == UnitTypeId.EXTRACTOR:
            anchor = cmd.target_pos or target
            geyser = self._nearest_geyser(anchor)
            if geyser is None:
                self._last_build_attempt_failed = True
                return True
            self.do(worker.build_gas(geyser))
            return True

        if cmd.unit_type_id is not None:
            self.do(worker.build(cmd.unit_type_id, target))
        return True

    async def _apply_train(self, cmd: Command, worker) -> bool:
        del worker
        if cmd.unit_type_id is None:
            self._last_build_attempt_failed = True
            return True
        pool = self.structures(UnitTypeId.SPAWNINGPOOL).ready
        if pool.exists:
            self.do(pool.first.train(cmd.unit_type_id))
        else:
            self._last_build_attempt_failed = True
        return True

    def decode_placement(self, cell_idx: int, center: Point2) -> Point2:
        """Map 16x16 cell index to world coordinates centered on manager target."""
        res = BUILD_MAP_RESOLUTION
        gx = int(cell_idx) % res
        gy = int(cell_idx) // res
        half = res // 2
        world_x = float(center.x) + (gx - half) + 0.5
        world_y = float(center.y) + (gy - half) + 0.5
        return Point2((world_x, world_y))

    def _nearest_geyser(self, pos: Point2):
        if not self.vespene_geyser:
            return None
        return self.vespene_geyser.closest_to(pos)

    def get_obs_focus_unit_tag(self) -> int:
        return int(self._focus_worker_tag)

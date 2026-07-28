"""Zerg build-agent bot: spawn, observations, actuation, rewards."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sc2.bot_ai import BotAI
from sc2.data import Race
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
# Standard melee opening bank; treat as OK when spawn.minerals == 0 (no debug grant).
_SC2_DEFAULT_START_MINERALS = 50


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
        self._train_baseline = 0
        self._commands_completed = 0
        self._mine_anchor_pos: Optional[Point2] = None

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
        if controller_name in {"scripted", "speed"}:
            if controller_name == "speed":
                from src.runtime.speed_mining_controller import SpeedMiningController

                return SpeedMiningController()
            from src.runtime.mining_controller import ScriptedMiningController

            return ScriptedMiningController()
        return NoOpMiningController()

    async def reset_game_state(self, verify: bool = True, skip_cleanup: bool = False):
        await self._ensure_map_visible()
        if not skip_cleanup:
            # Zero mineral/gas bank via map Galaxy trigger (chat SOFT_RESET).
            await self._request_map_soft_reset_resources()
            await self._clear_debug_spawned_neutrals()
            await self._clear_map_units(strict_empty_map=True)
            await self._ensure_own_units_cleared()
            # Bank assert not needed when the map trigger clears resources; hard
            # remakes still run every env.every_k_episodes (and when patches die).
        self._clear_spawn_session_state()
        self.worker_registry.clear()
        self.command_queue.clear()
        self.scripted_manager.reset()
        clear = getattr(self.mining_controller, "clear", None)
        if callable(clear):
            clear()
        self.episode_step = 0
        self._last_build_attempt_failed = False
        self._train_baseline = 0
        self._commands_completed = 0
        self._mine_anchor_pos = None
        await self._spawn_scenario()
        self._structures_at_step_start = len(self.structures)
        self._patch_script_targets()
        # Completion-gated script: emit MINE (standing) then the first macro only.
        # Later steps (e.g. TRAIN count=6) appear via emit_next_if_idle when idle.
        max_emits = len(self.scripted_manager.script_steps) + 1
        for _ in range(max_emits):
            if not self.scripted_manager.emit_next_if_idle():
                break
            if self.command_queue.active is not None:
                break
        # Training default: every worker mines near hatch if manager omitted MINE.
        self._ensure_training_mine_command()
        self._apply_standing_mine_command()
        self._sync_train_baseline()
        active = self.command_queue.active
        if active and active.unit_type_id is not None:
            level = self.training_conf.get("level_name", "?")
            count = getattr(active, "count", None)
            count_s = f" count={count}" if count else ""
            logger.info(
                f"Build level '{level}' macro command: "
                f"{active.command_type.name} {active.unit_type_id.name}{count_s} "
                f"target={active.target_pos} "
                f"(pending={len(self.command_queue.peek_pending())})"
            )
        await self._kick_economy_orders()

    async def _request_map_soft_reset_resources(self) -> None:
        """Ask the map Galaxy SoftReset trigger to zero minerals/gas via chat.

        Maps under ``train/maps/ladder/build_agent`` must define a SoftReset
        trigger on exact chat ``SOFT_RESET``. No-op if chat is unavailable.
        """
        client = getattr(self, "client", None)
        if client is None or not hasattr(client, "chat_send"):
            return
        try:
            await self.chat_send("SOFT_RESET")
        except Exception as exc:
            logger.warning(f"SOFT_RESET chat failed (map trigger may be missing): {exc}")
            return
        # Give Galaxy a couple of simulation steps to apply PlayerModifyPropertyInt.
        for _ in range(2):
            try:
                await client.step()
            except Exception:
                break
        try:
            await self._refresh_state()
        except Exception:
            pass

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
        self._mine_anchor_pos = None

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
        await self._ensure_own_units_cleared()

    async def _ensure_own_units_cleared(self) -> None:
        """Kill all remaining self-owned units/structures after the main cleanup loop."""
        await self._refresh_state()
        remaining = self.all_units.filter(self._unit_is_build_scenario_cleanup)
        if not remaining:
            return
        await self.client.debug_kill_unit([int(u.tag) for u in remaining])
        for _ in range(2):
            await self.client.step()
        await self._refresh_state()
        leftover = self.all_units.filter(self._unit_is_build_scenario_cleanup)
        if leftover:
            raise MapCleanupFailedError(leftover.amount)

    def _assert_banked_minerals_match_spawn(self) -> None:
        """Legacy helper kept for tests; soft resets no longer call this.

        SC2 cannot subtract the mineral bank. Soft resets therefore tolerate a
        leftover bank; ``env.every_k_episodes`` schedules hard remakes that
        restore opening resources and mineral patches.
        """
        target = int(self._spawn_config().get("minerals", 0))
        current = int(getattr(self, "minerals", 0) or 0)
        if target == 0 and current > _SC2_DEFAULT_START_MINERALS:
            logger.debug(
                f"Soft reset with leftover mineral bank={current} "
                f"(hard remake via every_k_episodes clears this)."
            )

    def _patch_script_targets(self):
        """Replace placeholder YAML targets with the hatch/spawn location.

        The management agent only provides a coarse region; ``[-1, -1]`` in YAML
        means \"use spawn/hatch\" (not a real map coordinate). The build policy
        then refines placement on the local 16x16 crop around that point.
        """
        hatch = (
            self.townhalls.first.position
            if self.townhalls.exists
            else (self._spawn_hatchery_pos or self.start_location)
        )
        patchable = {
            CommandType.BUILD,
            CommandType.EXPAND,
            CommandType.MINE,
        }
        for step in self.scripted_manager.script_steps:
            cmd = step.command
            if cmd.target_pos is None:
                if cmd.command_type in patchable:
                    cmd.target_pos = hatch
                continue
            if cmd.target_pos.x == -1 and cmd.target_pos.y == -1:
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
            logger.warning(
                f"Build map cleanup incomplete ({n} unit(s) left). Types: {dict(type_hist)}"
            )
            raise MapCleanupFailedError(n)

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
        """Fallback anchor only when no expansion/start locations exist."""
        start = getattr(self, "start_location", None)
        if start is not None:
            return start
        if self.game_info is not None:
            starts = list(getattr(self.game_info, "start_locations", None) or [])
            if starts:
                return starts[0]
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

    def _candidate_base_locations(self) -> List[Point2]:
        """Official map start locations (mains), not every expansion on the map.

        ``no_trigger`` AIE maps still expose ``game_info.start_locations``; removing
        triggers does not strip LocStartLoc metadata. Prefer those so the hatch is
        at a real spawn, not a random natural/third base from
        ``expansion_locations_list``.
        """
        locs: List[Point2] = []
        if self.game_info is not None:
            starts = list(getattr(self.game_info, "start_locations", None) or [])
            locs.extend(starts)
        # burnysc2 also mirrors starts on the bot (excluding our own when set).
        for p in list(getattr(self, "enemy_start_locations", None) or []):
            locs.append(p)
        own = getattr(self, "start_location", None)
        if own is not None:
            locs.append(own)

        if not locs:
            # Last resort: expansions (blank grids / maps without LocStartLoc).
            try:
                locs.extend(list(getattr(self, "expansion_locations_list", None) or []))
            except Exception as exc:
                logger.debug(f"expansion_locations_list unavailable: {exc}")

        seen = set()
        unique: List[Point2] = []
        for p in locs:
            key = (round(float(p.x), 1), round(float(p.y), 1))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique

    def _mineral_amount(self, mineral) -> int:
        for attr in ("mineral_contents", "health", "energy"):
            val = getattr(mineral, attr, None)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
        return 1500

    def _base_has_healthy_minerals(
        self,
        pos: Point2,
        radius: float = 12.0,
        min_patches: int = 4,
        min_total: int = 1000,
    ) -> bool:
        fields = [m for m in self.mineral_field if m.distance_to(pos) <= radius]
        if len(fields) < min_patches:
            return False
        total = sum(self._mineral_amount(m) for m in fields)
        return total >= min_total

    def _pick_episode_base_pos(self, rng: Optional[random.Random] = None) -> Point2:
        """
        Random healthy official start location for this episode.

        Raises MapCleanupFailedError when starts exist but none still have healthy
        mineral fields (soft reset cannot refill patches → Level0Env hard remake).
        """
        rng = rng or random
        candidates = self._candidate_base_locations()
        if not candidates:
            logger.warning(
                "No start_locations on map; falling back to pathable hatch placement."
            )
            return self._default_hatch_location()

        healthy = [c for c in candidates if self._base_has_healthy_minerals(c)]
        if not healthy:
            raise MapCleanupFailedError(len(candidates))

        pool = list(healthy)
        rng.shuffle(pool)
        hatch_pos = pool[0]
        if not self._is_pathable(hatch_pos):
            hatch_pos = self._find_buildable_near(hatch_pos, min_dist=0.0, max_dist=4.0)
        if self.vespene_geyser.exists:
            self._target_geyser_pos = self.vespene_geyser.closest_to(hatch_pos).position
        else:
            self._target_geyser_pos = None
        return hatch_pos

    def _resolve_hatch_spawn_pos(self) -> Point2:
        """Pick a fresh random optimal base each episode (no cross-episode reuse)."""
        hatch_pos = self._pick_episode_base_pos()
        self._cached_hatch_spawn_pos = hatch_pos
        self._cached_spawn_geometry_key = (
            f"{getattr(self.game_info, 'map_name', 'unknown')}:"
            f"{round(hatch_pos.x, 1)}:{round(hatch_pos.y, 1)}"
        )
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
        num_drones = int(spawn.get("drones", 8))
        minerals = int(spawn.get("minerals", 0))
        vespene = int(spawn.get("vespene", 0))
        num_overlords = int(spawn.get("overlords", 0))
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

        # Pre-spawn overlords for supply (scenario setup; not agent help).
        for i in range(num_overlords):
            opos = Point2((hatch_pos.x - 3.0 - i, hatch_pos.y + 3.0))
            await self.client.debug_create_unit(
                [[UnitTypeId.OVERLORD, 1, opos, player]]
            )

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
        """Legacy helper — prefer ``_apply_standing_mine_command`` / MINE commands."""
        spawn = self._spawn_config()
        drones_sorted = sorted(self.workers, key=lambda u: u.tag)
        num_drones = len(drones_sorted)
        mining_count = int(spawn.get("mining_drones", 0))

        if mining_count <= 0 and self._mining_controller_name() in {"scripted", "speed"}:
            mining_count = num_drones

        mining_count = min(max(0, mining_count), num_drones)

        for i, drone in enumerate(drones_sorted):
            role = WorkerRole.MINING if i < mining_count else WorkerRole.BUILD
            self.worker_registry.assign(drone.tag, role)

    def _closest_hatch_to(self, pos: Point2) -> Point2:
        if self.townhalls.exists:
            return self.townhalls.closest_to(pos).position
        if self._spawn_hatchery_pos is not None:
            return self._spawn_hatchery_pos
        return pos

    def _ensure_training_mine_command(self) -> None:
        """If the manager script omitted MINE, assign every worker to mine at hatch."""
        if self.command_queue.standing_mine is not None:
            return
        if self._mining_controller_name() not in {"scripted", "speed"}:
            return
        hatch = (
            self.townhalls.first.position
            if self.townhalls.exists
            else self._spawn_hatchery_pos
        )
        if hatch is None:
            return
        n_workers = len(list(self.workers))
        cmd = Command(
            command_type=CommandType.MINE,
            target_pos=hatch,
            worker_count=n_workers if n_workers > 0 else None,
        )
        self.command_queue.enqueue(cmd)

    def _apply_standing_mine_command(self) -> None:
        """Resolve MINE: N drones nearest the region → MINING at closest hatch."""
        cmd = self.command_queue.standing_mine
        if cmd is None or cmd.command_type != CommandType.MINE:
            return
        region = cmd.target_pos or self._spawn_hatchery_pos
        if region is None:
            return
        hatch = self._closest_hatch_to(region)
        self._mine_anchor_pos = hatch

        workers = list(self.workers)
        if not workers:
            return

        if cmd.assigned_unit_tags:
            selected = [w for w in workers if int(w.tag) in cmd.assigned_unit_tags]
        else:
            n = cmd.worker_count
            if n is None or n <= 0:
                n = len(workers)
            ordered = sorted(workers, key=lambda w: w.distance_to(region))
            selected = ordered[: min(int(n), len(ordered))]

        selected_tags = {int(w.tag) for w in selected}
        for worker in workers:
            tag = int(worker.tag)
            if tag in selected_tags:
                self.worker_registry.assign(tag, WorkerRole.MINING)
            elif self.worker_registry.get_role(tag) == WorkerRole.MINING:
                # Drop miners not covered by this MINE (pulled off the line).
                self.worker_registry.remove(tag)

        cmd.assigned_unit_tags = selected_tags

    def action_masks(self) -> np.ndarray:
        """Mask padded workers; for BUILD, restrict placement to buildable cells."""
        mask = np.ones(MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS, dtype=bool)
        n_workers = len(list(self.workers))
        if n_workers < MAX_BUILD_WORKERS:
            mask[n_workers:MAX_BUILD_WORKERS] = False
        placement = self._placement_action_mask()
        mask[MAX_BUILD_WORKERS:] = placement
        return mask

    def _placement_action_mask(self) -> np.ndarray:
        """Legal 16x16 cells near the manager coarse target (creep+path for pool)."""
        out = np.ones(BUILD_PLACEMENT_CELLS, dtype=bool)
        cmd = self.command_queue.active
        if cmd is None or cmd.command_type == CommandType.TRAIN:
            return out
        if cmd.command_type not in {CommandType.BUILD, CommandType.EXPAND}:
            return out
        center = cmd.target_pos or self._spawn_hatchery_pos
        if center is None:
            return out

        needs_creep = cmd.unit_type_id not in {
            UnitTypeId.HATCHERY,
            UnitTypeId.LAIR,
            UnitTypeId.HIVE,
            UnitTypeId.EXTRACTOR,
        }
        legal = np.zeros(BUILD_PLACEMENT_CELLS, dtype=bool)
        for idx in range(BUILD_PLACEMENT_CELLS):
            pos = self.decode_placement(idx, center)
            x, y = int(pos.x), int(pos.y)
            if not self._grid_pathable(x, y):
                continue
            if needs_creep and not self._cell_has_creep(x, y):
                continue
            legal[idx] = True
        # Avoid a fully-masked spatial head (MaskablePPO cannot sample).
        if not legal.any():
            return out
        return legal

    def _cell_has_creep(self, x: int, y: int) -> bool:
        creep = getattr(getattr(self, "state", None), "creep", None)
        if creep is None:
            return True
        try:
            if hasattr(creep, "is_set"):
                return bool(creep.is_set(Point2((float(x), float(y)))))
            data = getattr(creep, "data_numpy", creep)
            if hasattr(data, "shape") and len(data.shape) == 2:
                h, w = data.shape
                if 0 <= x < w and 0 <= y < h:
                    return bool(data[y, x])
        except Exception:
            return True
        return True

    async def on_build_substep(self):
        """Per game-step runtime: manager emit-if-idle, mining, command activation."""
        self.scripted_manager.on_game_step()
        self.scripted_manager.emit_next_if_idle()
        self.command_queue.activate_next()
        if self.command_queue.standing_mine is not None:
            self._apply_standing_mine_command()
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
        if self._try_complete_active_command():
            reward += float(self.rewards_cfg.get("command_complete", 15.0))
            # Promote next scripted manager intent (e.g. TRAIN batch after BUILD).
            self.scripted_manager.emit_next_if_idle()
            self._sync_train_baseline()
        done = self._episode_complete()
        if done:
            reward += float(self.rewards_cfg.get("success", 100.0))
        return obs, reward, done, False, info

    def _compute_reward(self) -> float:
        r = float(self.rewards_cfg.get("step_penalty", -0.05))
        if self._last_build_attempt_failed:
            r += float(self.rewards_cfg.get("invalid_build_penalty", -1.0))
        r += self._compute_mining_reward()
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

    def _mining_role_workers(self) -> List:
        tags = self.worker_registry.mining_tags()
        return [u for u in self.workers if int(u.tag) in tags]

    def _compute_mining_reward(self) -> float:
        """Reward miners that stay gathering/returning; penalize idle mining-role drones."""
        active_bonus = float(self.rewards_cfg.get("mining_active_bonus", 0.0))
        idle_penalty = float(self.rewards_cfg.get("idle_miner_penalty", 0.0))
        if active_bonus == 0.0 and idle_penalty == 0.0:
            return 0.0
        r = 0.0
        for worker in self._mining_role_workers():
            carrying = bool(
                getattr(worker, "is_carrying_minerals", False)
                or getattr(worker, "is_carrying_vespene", False)
            )
            gathering = bool(getattr(worker, "is_gathering", False))
            idle = bool(getattr(worker, "is_idle", False))
            if gathering or carrying:
                r += active_bonus
            elif idle:
                r += idle_penalty
        return r

    def _unit_count(self, unit_type: UnitTypeId) -> int:
        return int(self.units(unit_type).amount)

    @staticmethod
    def _train_target_count(cmd: Command) -> int:
        """How many units a TRAIN command must produce (None/<=0 => 1)."""
        raw = getattr(cmd, "count", None)
        if raw is None or int(raw) <= 0:
            return 1
        return int(raw)

    def _sync_train_baseline(self) -> None:
        cmd = self.command_queue.active
        if (
            cmd is not None
            and cmd.command_type == CommandType.TRAIN
            and cmd.unit_type_id is not None
        ):
            self._train_baseline = self._unit_count(cmd.unit_type_id)
        else:
            self._train_baseline = 0

    def _try_complete_active_command(self) -> bool:
        """Complete the active macro command when its goal is met. Not episode done."""
        cmd = self.command_queue.active
        if cmd is None:
            return False
        completed = False
        if cmd.command_type in {CommandType.BUILD, CommandType.EXPAND}:
            if cmd.unit_type_id is not None and self.structures(cmd.unit_type_id).ready.exists:
                completed = True
        elif cmd.command_type == CommandType.TRAIN:
            if cmd.unit_type_id is not None:
                target = self._train_target_count(cmd)
                if self._unit_count(cmd.unit_type_id) >= self._train_baseline + target:
                    completed = True
        if not completed:
            return False
        self.command_queue.complete_active()
        self._commands_completed += 1
        self.command_queue.activate_next()
        self.scripted_manager.emit_next_if_idle()
        self._sync_train_baseline()
        return True

    def _episode_complete(self) -> bool:
        """True when the full manager script has been finished."""
        script_done = (
            self.scripted_manager._script_index
            >= len(self.scripted_manager.script_steps)
        )
        return (
            script_done
            and self.command_queue.active is None
            and not self.command_queue.peek_pending()
            and self._commands_completed > 0
        )

    async def apply_build_action(self, action: np.ndarray) -> bool:
        """Decode policy action into SC2 build/train order."""
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

        return await self._apply_build(cmd, worker, placement_idx)

    async def _apply_build(self, cmd: Command, worker, placement_idx: int) -> bool:
        """Agent-faithful build: issue worker.build at the decoded placement cell."""
        self._last_build_attempt_failed = False
        if cmd.unit_type_id is None:
            self._last_build_attempt_failed = True
            return True

        # Already building / built — wait for _try_complete_active_command.
        if self.structures(cmd.unit_type_id).exists:
            return True
        try:
            if self.already_pending(cmd.unit_type_id) > 0:
                return True
        except Exception:
            pass

        if not self.can_afford(cmd.unit_type_id):
            # Waiting for scripted mining to bank minerals — not an invalid attempt.
            return True

        target = self.decode_placement(
            placement_idx, cmd.target_pos or worker.position
        )
        # Pull selected miner off the mining controller for this build order.
        self.worker_registry.set_role(worker.tag, WorkerRole.BUILD)
        self.do(worker.build(cmd.unit_type_id, target))
        return True

    async def _apply_train(self, cmd: Command, worker) -> bool:
        """Thin TRAIN helper: morph larva until baseline+count is queued/produced."""
        del worker
        self._last_build_attempt_failed = False
        if cmd.unit_type_id is None:
            self._last_build_attempt_failed = True
            return True
        if not self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists:
            return True
        target = self._train_target_count(cmd)
        current = self._unit_count(cmd.unit_type_id)
        try:
            pending = int(self.already_pending(cmd.unit_type_id))
        except Exception:
            pending = 0
        # Stop issuing morphs once enough units are alive or already in production.
        if current + pending >= self._train_baseline + target:
            return True
        if not self.can_afford(cmd.unit_type_id):
            return True
        larva = self.larva
        if not larva.exists:
            return True
        self.do(larva.random.train(cmd.unit_type_id))
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

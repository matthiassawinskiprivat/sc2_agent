"""Zerg build-agent bot: spawn, observations, actuation, rewards."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from sc2.bot_ai import BotAI
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.sc2_bot import (
    MapCleanupFailedError,
    alliance_value,
    unit_is_subject_to_map_cleanup,
)
from src.agent.build_constants import (
    BUILD_MAP_RESOLUTION,
    BUILD_PLACEMENT_CELLS,
    MAX_BUILD_WORKERS,
)
from src.agent.commands import INJECT_ENERGY_COST, Command, CommandType
from src.runtime.command_queue import CommandQueue
from src.runtime.mining_controller import MiningController, NoOpMiningController
from src.runtime.scripted_manager import ManagerScriptStep, ScriptedManager
from src.runtime.squad_registry import SquadRegistry
from src.runtime.stub_combat_controller import StubCombatController
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
            self.scripted_manager = ScriptedManager.from_level_config(
                env_cfg, can_emit_step=self._can_emit_script_step
            )
            self.scripted_manager.worker_registry = self.worker_registry
            self.scripted_manager.command_queue = self.command_queue
        else:
            self.scripted_manager = ScriptedManager(
                script_steps=[],
                macro_tick_interval=int(env_cfg.get("macro_tick_interval", 8)),
                worker_registry=self.worker_registry,
                command_queue=self.command_queue,
                can_emit_step=self._can_emit_script_step,
            )
        # Remember which script steps used hatch placeholders ([-1,-1] / None).
        # Command objects are mutated in place; without this, episode 2+ keep the
        # previous hatch coordinates and placement desyncs from the new base.
        self._script_anchor_to_hatch = self._compute_script_hatch_anchors()
        self.obs_builder = BuildObservationBuilder(self.training_conf)
        self.mining_controller: MiningController = self._resolve_mining_controller()
        self.stub_combat = StubCombatController()
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
        self._gas_worker_quota: int = 0
        self._gas_anchor_pos: Optional[Point2] = None
        self._inject_succeeded = False
        self._keep_enemy_alive = bool(
            self.training_conf.get("keep_enemy_alive")
            or (self.training_conf.get("env") or {}).get("keep_enemy_alive")
            or str(self.training_conf.get("level_name", "")).startswith("level_manager_")
        )

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
            # Bank assert not needed when the map trigger clears resources; hard
            # remakes still run every env.every_k_episodes (and when patches die).
        # Always clear trainee + Computer seat-filler units (ladder openers / AI).
        await self._clear_map_units(strict_empty_map=not skip_cleanup)
        if not skip_cleanup:
            await self._ensure_own_units_cleared()
        self._clear_spawn_session_state()
        self.worker_registry.clear()
        self.command_queue.clear()
        self.scripted_manager.reset()
        # Refresh hatch-anchor flags from restored pristine YAML sentinels.
        self._script_anchor_to_hatch = self._compute_script_hatch_anchors()
        clear = getattr(self.mining_controller, "clear", None)
        if callable(clear):
            clear()
        self.episode_step = 0
        self._last_build_attempt_failed = False
        self._train_baseline = 0
        self._commands_completed = 0
        self._inject_succeeded = False
        self._mine_anchor_pos = None
        self._gas_worker_quota = 0
        self._gas_anchor_pos = None
        await self._spawn_scenario()
        # Computer may have respawned during spawn steps — wipe again.
        await self._suppress_seat_filler()
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
        self._apply_standing_mine_gas_command()
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

    def _compute_script_hatch_anchors(self) -> List[bool]:
        """True for script steps whose target should track the current hatch.

        Prefer pristine YAML templates so soft-reset rebinding still works after
        working ``script_steps`` have been patched to real map coordinates.
        """
        pristine = getattr(self.scripted_manager, "_pristine_steps", None)
        steps = pristine if pristine is not None else self.scripted_manager.script_steps
        patchable = {
            CommandType.BUILD,
            CommandType.EXPAND,
            CommandType.MINE,
            CommandType.MINE_GAS,
        }
        flags: List[bool] = []
        for step in steps:
            cmd = step.command
            if cmd.command_type not in patchable:
                flags.append(False)
                continue
            tp = cmd.target_pos
            if tp is None:
                flags.append(True)
            elif float(tp.x) == -1.0 and float(tp.y) == -1.0:
                flags.append(True)
            else:
                flags.append(False)
        return flags

    def _patch_script_targets(self):
        """Bind hatch-anchored script targets to the current hatch/spawn location.

        The management agent only provides a coarse region; ``[-1, -1]`` in YAML
        means \"use spawn/hatch\" (not a real map coordinate). The build policy
        then refines placement on the local 16x16 crop around that point.

        EXTRACTOR is special: the crop centers on the **local geyser** next to the
        episode hatch (not the hatch footprint, and never a far-side geyser).

        Must re-run every episode. ``ScriptedManager.reset`` restores pristine
        YAML sentinels, then this rebinds hatch/geyser for the new spawn.
        Always recompute the local geyser from the current hatch.
        """
        hatch = self._episode_hatch_pos()
        if hatch is None:
            return
        # Anchors come from pristine YAML sentinels (set at init / after reset),
        # never from already-patched coordinates.
        anchors = getattr(self, "_script_anchor_to_hatch", None)
        if anchors is None:
            anchors = self._compute_script_hatch_anchors()
            self._script_anchor_to_hatch = anchors

        # Force a fresh geyser for this hatch (ignore any leftover coords).
        local_geyser = self._force_local_geyser_pos(hatch)

        for i, step in enumerate(self.scripted_manager.script_steps):
            cmd = step.command
            anchored = i < len(anchors) and anchors[i]
            # Always rebind EXTRACTOR from this episode's hatch geyser.
            if (
                cmd.command_type == CommandType.BUILD
                and cmd.unit_type_id
                in {UnitTypeId.EXTRACTOR, UnitTypeId.EXTRACTORRICH}
            ):
                cmd.target_pos = local_geyser if local_geyser is not None else hatch
                continue
            if anchored:
                cmd.target_pos = hatch
                continue
            # Legacy / dynamic scripts still using the placeholder sentinel.
            tp = cmd.target_pos
            if tp is not None and float(tp.x) == -1.0 and float(tp.y) == -1.0:
                cmd.target_pos = hatch

    def _episode_hatch_pos(self) -> Optional[Point2]:
        """Prefer this episode's spawned hatch — never an arbitrary townhalls.first."""
        if self._spawn_hatchery_pos is not None:
            return self._spawn_hatchery_pos
        if self.townhalls and self.townhalls.exists:
            return self.townhalls.first.position
        return getattr(self, "start_location", None)

    def _force_local_geyser_pos(self, hatch: Point2) -> Optional[Point2]:
        """Recompute the episode geyser from ``hatch`` (never keep a stale point)."""
        geyser_unit = self._local_geyser_unit(hatch)
        if geyser_unit is None:
            self._target_geyser_pos = None
            return None
        self._target_geyser_pos = geyser_unit.position
        return self._target_geyser_pos

    def _resolve_local_geyser_pos(self, hatch: Point2) -> Optional[Point2]:
        """Geyser used for EXTRACTOR / MINE_GAS near the episode hatch."""
        return self._force_local_geyser_pos(hatch)

    def _local_geyser_unit(self, near: Point2, max_dist: float = 20.0):
        """Closest geyser within ``max_dist`` of ``near`` (None if none local)."""
        geysers = getattr(self, "vespene_geyser", None)
        if not geysers:
            return None
        if hasattr(geysers, "exists") and not geysers.exists:
            return None
        try:
            closest = geysers.closest_to(near)
        except Exception:
            return None
        try:
            dist = float(closest.distance_to(near))
        except (TypeError, ValueError):
            return closest
        if dist > max_dist:
            return None
        return closest

    def _unit_is_build_scenario_cleanup(self, u) -> bool:
        """Remove trainee + Computer seat-filler units; keep neutrals/resources."""
        if not unit_is_subject_to_map_cleanup(u):
            return False
        from sc2.data import Alliance

        av = alliance_value(getattr(u, "alliance", None))
        if av is None:
            return False
        # Self=1, Enemy=4. Neutral=3 must never be killed (minerals/geysers).
        if av == Alliance.Enemy.value and self._keep_enemy_alive:
            return False
        return av in {Alliance.Self.value, Alliance.Enemy.value}

    async def _clear_map_units(self, strict_empty_map: bool = False):
        """Kill trainee + seat-filler units so resets do not stack hatcheries."""
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
        if remaining and strict_empty_map:
            n = remaining.amount
            type_hist = Counter(
                getattr(u.type_id, "name", str(u.type_id)) for u in remaining
            )
            logger.warning(
                f"Build map cleanup incomplete ({n} unit(s) left). Types: {dict(type_hist)}"
            )
            raise MapCleanupFailedError(n)

    async def _suppress_seat_filler(self) -> None:
        """Keep Computer opponent from interfering with the build curriculum.

        Manager levels with a live Computer must not wipe the enemy every step.
        """
        if self._keep_enemy_alive:
            return
        tags = []
        for group in (getattr(self, "enemy_units", None), getattr(self, "enemy_structures", None)):
            if not group:
                continue
            tags.extend(int(u.tag) for u in group)
        if not tags:
            return
        await self.client.debug_kill_unit(tags)

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
        """Optimal townhall spots: expansion locations (mineral/gas clusters).

        Prefer burnysc2 ``expansion_locations_list`` — these are the standard
        base centers as close to minerals as the map layout allows. Fall back to
        LocStartLoc mains, then mineral-cluster centroids.
        """
        locs: List[Point2] = []
        try:
            locs.extend(list(getattr(self, "expansion_locations_list", None) or []))
        except Exception as exc:
            logger.debug(f"expansion_locations_list unavailable: {exc}")
        if not locs:
            try:
                exp_dict = getattr(self, "expansion_locations_dict", None) or {}
                locs.extend(list(exp_dict.keys()))
            except Exception as exc:
                logger.debug(f"expansion_locations_dict unavailable: {exc}")

        if not locs:
            if self.game_info is not None:
                locs.extend(list(getattr(self.game_info, "start_locations", None) or []))
            for p in list(getattr(self, "enemy_start_locations", None) or []):
                locs.append(p)
            own = getattr(self, "start_location", None)
            if own is not None:
                locs.append(own)

        if not locs:
            locs.extend(self._bases_from_mineral_clusters())

        seen = set()
        unique: List[Point2] = []
        for p in locs:
            key = (round(float(p.x), 1), round(float(p.y), 1))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique

    def _bases_from_mineral_clusters(
        self, cluster_radius: float = 10.0
    ) -> List[Point2]:
        """Derive base centers from mineral patch clusters when expansions missing."""
        fields = list(self.mineral_field) if self.mineral_field else []
        if not fields:
            return []
        unused = set(range(len(fields)))
        centers: List[Point2] = []
        while unused:
            seed = unused.pop()
            cluster = [seed]
            queue = [seed]
            while queue:
                j = queue.pop()
                j_pos = fields[j].position
                for i in list(unused):
                    if fields[i].distance_to(j_pos) <= cluster_radius:
                        unused.remove(i)
                        cluster.append(i)
                        queue.append(i)
            if len(cluster) < 4:
                continue
            members = [fields[i] for i in cluster]
            cx = sum(float(m.position.x) for m in members) / len(members)
            cy = sum(float(m.position.y) for m in members) / len(members)
            mineral_center = Point2((cx, cy))
            # Offset hatch toward map center so it sits in front of the mineral line.
            map_center = getattr(
                getattr(self, "game_info", None), "map_center", None
            )
            if map_center is not None:
                toward_x = float(map_center.x) - mineral_center.x
                toward_y = float(map_center.y) - mineral_center.y
                norm = math.hypot(toward_x, toward_y) or 1.0
                hatch = Point2(
                    (
                        mineral_center.x + 5.5 * toward_x / norm,
                        mineral_center.y + 5.5 * toward_y / norm,
                    )
                )
            else:
                hatch = mineral_center
            centers.append(hatch)
        return centers

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

    @staticmethod
    def _half_tile(pos: Point2) -> Point2:
        """Townhall centers sit on (*.5, *.5) because the footprint is 5×5."""
        return Point2((math.floor(float(pos.x)) + 0.5, math.floor(float(pos.y)) + 0.5))

    def _map_start_locations(self) -> List[Point2]:
        """Exact LocStartLoc positions from the map (perfect main-base hatch spots)."""
        locs: List[Point2] = []
        if self.game_info is not None:
            locs.extend(list(getattr(self.game_info, "start_locations", None) or []))
        for p in list(getattr(self, "enemy_start_locations", None) or []):
            locs.append(p)
        own = getattr(self, "start_location", None)
        if own is not None:
            locs.append(own)
        seen = set()
        unique: List[Point2] = []
        for p in locs:
            key = (round(float(p.x), 1), round(float(p.y), 1))
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique

    def _resources_for_base(self, base_pos: Point2) -> List:
        """Mineral/gas units belonging to this base (expansion dict, else radius)."""
        try:
            exp_dict = getattr(self, "expansion_locations_dict", None) or {}
            if exp_dict:
                nearest = min(exp_dict.keys(), key=lambda p: p.distance_to(base_pos))
                if nearest.distance_to(base_pos) <= 8.0:
                    resources = list(exp_dict[nearest])
                    if resources:
                        return resources
        except Exception as exc:
            logger.debug(f"expansion resource lookup failed: {exc}")
        minerals = [
            m for m in (self.mineral_field or []) if m.distance_to(base_pos) <= 12.0
        ]
        geysers = [
            g
            for g in (list(self.vespene_geyser) if self.vespene_geyser else [])
            if g.distance_to(base_pos) <= 12.0
        ]
        return minerals + geysers

    def _refine_hatch_on_placement_grid(self, resources: Sequence) -> Optional[Point2]:
        """Burnysc2-style search: closest legal (*.5,*.5) townhall to the resource pack."""
        if not resources or self.game_info is None:
            return None
        placement = getattr(self.game_info, "placement_grid", None)
        if placement is None:
            return None
        try:
            from sc2.constants import geyser_ids
        except Exception:
            geyser_ids = set()

        amount = len(resources)
        center_x = int(sum(float(r.position.x) for r in resources) / amount) + 0.5
        center_y = int(sum(float(r.position.y) for r in resources) / amount) + 0.5
        offset_range = 7
        offsets = [
            (x, y)
            for x, y in itertools.product(
                range(-offset_range, offset_range + 1), repeat=2
            )
            if 4 < math.hypot(x, y) <= 8
        ]
        possible: List[Point2] = []
        for ox, oy in offsets:
            point = Point2((center_x + ox, center_y + oy))
            try:
                if int(placement[point.rounded]) != 1:
                    continue
            except Exception:
                continue
            legal = True
            for resource in resources:
                min_dist = 6.0
                try:
                    ut = resource._proto.unit_type
                    if ut in geyser_ids:
                        min_dist = 7.0
                except Exception:
                    name = str(getattr(resource, "name", "") or "")
                    if "Geyser" in name or "Vespene" in name:
                        min_dist = 7.0
                if point.distance_to(resource) < min_dist:
                    legal = False
                    break
            if legal:
                possible.append(point)
        if not possible:
            return None
        return min(
            possible, key=lambda p: sum(p.distance_to(r) for r in resources)
        )

    def _optimal_hatch_at_base(self, base_pos: Point2) -> Point2:
        """Exact main LocStartLoc, else placement-grid refine, else half-tile snap."""
        # 1) Map-authored start locations beat computed expansions for mains.
        for start in self._map_start_locations():
            if start.distance_to(base_pos) <= 5.0:
                return self._half_tile(start)

        # 2) Recompute the burnysc2 optimal townhall against this base's resources.
        resources = self._resources_for_base(base_pos)
        refined = self._refine_hatch_on_placement_grid(resources)
        if refined is not None:
            return refined

        # 3) Fall back to expansion dict key / half-tile of the candidate.
        try:
            exp_dict = getattr(self, "expansion_locations_dict", None) or {}
            if exp_dict:
                nearest = min(exp_dict.keys(), key=lambda p: p.distance_to(base_pos))
                if nearest.distance_to(base_pos) <= 6.0:
                    return self._half_tile(nearest)
        except Exception as exc:
            logger.debug(f"expansion_locations_dict snap failed: {exc}")
        return self._half_tile(base_pos)

    def _pick_episode_base_pos(self, rng: Optional[random.Random] = None) -> Point2:
        """
        Random healthy optimal base for this episode (near minerals).

        Raises MapCleanupFailedError when bases exist but none still have healthy
        mineral fields (soft reset cannot refill patches → Level0Env hard remake).
        Never falls back to a random pathable tile elsewhere on the map.
        """
        rng = rng or random
        candidates = self._candidate_base_locations()
        if not candidates:
            raise MapCleanupFailedError(0)

        healthy = [c for c in candidates if self._base_has_healthy_minerals(c)]
        if not healthy:
            raise MapCleanupFailedError(len(candidates))

        pool = list(healthy)
        rng.shuffle(pool)
        hatch_pos = self._optimal_hatch_at_base(pool[0])
        if self.vespene_geyser.exists:
            self._target_geyser_pos = self.vespene_geyser.closest_to(hatch_pos).position
        else:
            self._target_geyser_pos = None
        return hatch_pos

    def _resolve_hatch_spawn_pos(self) -> Point2:
        """Pick a fresh optimal base each episode (rotates among healthy expansions)."""
        hatch_pos = self._pick_episode_base_pos()
        self._cached_hatch_spawn_pos = hatch_pos
        self._cached_spawn_geometry_key = (
            f"{getattr(self.game_info, 'map_name', 'unknown')}:"
            f"{round(hatch_pos.x, 1)}:{round(hatch_pos.y, 1)}"
        )
        logger.info(
            f"Build hatch spawn at optimal base "
            f"({hatch_pos.x:.1f}, {hatch_pos.y:.1f})"
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

    def _prune_missing_worker_roles(self) -> None:
        """Drop registry entries for drones that left (e.g. morphed into a building)."""
        live = {int(w.tag) for w in self.workers}
        for tag in list(self.worker_registry.snapshot().keys()):
            if tag not in live:
                self.worker_registry.remove(tag)

    def _release_idle_builders(self) -> None:
        """Return idle BUILD-role drones to mining (failed / cancelled build orders).

        While a BUILD/EXPAND macro is still unsatisfied (no structure / pending),
        keep BUILD roles so a one-frame idle observation cannot demote a traveler
        and let mining cancel the order.
        """
        from src.runtime.mining_controller import _worker_is_building

        cmd = self.command_queue.active
        if (
            cmd is not None
            and cmd.command_type in {CommandType.BUILD, CommandType.EXPAND}
            and cmd.unit_type_id is not None
            and not self._has_structure_near_episode_hatch(cmd.unit_type_id)
        ):
            try:
                if int(self.already_pending(cmd.unit_type_id)) == 0:
                    return
            except Exception:
                return

        for tag in list(self.worker_registry.build_pool_tags()):
            worker = next((w for w in self.workers if int(w.tag) == tag), None)
            if worker is None:
                self.worker_registry.remove(tag)
                continue
            if _worker_is_building(worker):
                continue
            if bool(getattr(worker, "is_idle", False)):
                self.worker_registry.assign(tag, WorkerRole.MINING)

    def _apply_standing_mine_command(self) -> None:
        """Resolve MINE: N drones nearest the region → MINING at closest hatch.

        Never overwrite BUILD/MICRO roles — otherwise speed-mining cancels an
        in-flight ``worker.build`` on the next substep.
        """
        cmd = self.command_queue.standing_mine
        if cmd is None or cmd.command_type != CommandType.MINE:
            return
        region = cmd.target_pos or self._spawn_hatchery_pos
        if region is None:
            return
        hatch = self._closest_hatch_to(region)
        self._mine_anchor_pos = hatch

        self._prune_missing_worker_roles()
        workers = list(self.workers)
        if not workers:
            return

        protected = {WorkerRole.BUILD, WorkerRole.MICRO}
        # Prefer free workers for the mine quota; leave builders alone.
        free = [
            w
            for w in workers
            if self.worker_registry.get_role(int(w.tag)) not in protected
        ]

        if cmd.assigned_unit_tags:
            # Keep sticky assignment only for workers still free / already mining.
            selected = [
                w
                for w in free
                if int(w.tag) in cmd.assigned_unit_tags
            ]
            if len(selected) < len(cmd.assigned_unit_tags):
                # Refill from nearest free workers if builders ate some slots.
                n = cmd.worker_count
                if n is None or n <= 0:
                    n = len(free)
                need = min(int(n), len(free)) - len(selected)
                if need > 0:
                    chosen = {int(w.tag) for w in selected}
                    ordered = sorted(
                        (w for w in free if int(w.tag) not in chosen),
                        key=lambda w: w.distance_to(region),
                    )
                    selected.extend(ordered[:need])
        else:
            n = cmd.worker_count
            if n is None or n <= 0:
                n = len(free)
            ordered = sorted(free, key=lambda w: w.distance_to(region))
            selected = ordered[: min(int(n), len(ordered))]

        selected_tags = {int(w.tag) for w in selected}
        for worker in workers:
            tag = int(worker.tag)
            role = self.worker_registry.get_role(tag)
            if role in protected:
                continue
            if tag in selected_tags:
                self.worker_registry.assign(tag, WorkerRole.MINING)
            elif role == WorkerRole.MINING:
                # Drop miners not covered by this MINE (pulled off the line).
                self.worker_registry.remove(tag)

        cmd.assigned_unit_tags = selected_tags

    def _apply_standing_mine_gas_command(self) -> None:
        """Resolve MINE_GAS: set gas worker quota for the mining controller.

        ``worker_count`` None / <=0 clears gas (pull drones off extractors).
        Does not change WorkerRegistry roles — gas workers stay MINING and are
        reserved inside the mining controller.
        """
        cmd = self.command_queue.standing_mine_gas
        if cmd is None or cmd.command_type != CommandType.MINE_GAS:
            self._gas_worker_quota = 0
            self._gas_anchor_pos = None
            return
        n = cmd.worker_count
        if n is None or int(n) <= 0:
            self._gas_worker_quota = 0
        else:
            self._gas_worker_quota = int(n)
        region = cmd.target_pos or self._spawn_hatchery_pos or self._mine_anchor_pos
        if region is not None:
            self._gas_anchor_pos = self._closest_hatch_to(region)
        else:
            self._gas_anchor_pos = self._mine_anchor_pos

    def action_masks(self) -> np.ndarray:
        """Vacuous mask for MaskablePPO — build agent does not gate actions."""
        return np.ones(MAX_BUILD_WORKERS + BUILD_PLACEMENT_CELLS, dtype=bool)

    def _structures_near_episode_hatch(
        self, unit_type: UnitTypeId, max_dist: float = 20.0
    ) -> list:
        """Own structures of ``unit_type`` near this episode's hatch (not leftovers)."""
        structs = self.structures(unit_type)
        if not structs:
            return []
        hatch = self._episode_hatch_pos()
        if hatch is None:
            return list(structs)
        local = []
        for s in structs:
            try:
                if float(s.distance_to(hatch)) <= max_dist:
                    local.append(s)
            except (TypeError, ValueError):
                local.append(s)
        return local

    def _has_structure_near_episode_hatch(
        self, unit_type: UnitTypeId, *, ready: bool = False, max_dist: float = 20.0
    ) -> bool:
        local = self._structures_near_episode_hatch(unit_type, max_dist=max_dist)
        if not local:
            return False
        if not ready:
            return True
        return any(bool(getattr(s, "is_ready", True)) for s in local)

    def _placement_cell_legal(self, cmd: Command, placement_idx: int) -> bool:
        """True if the decoded cell is pathable (and on creep when required)."""
        if cmd.command_type not in {CommandType.BUILD, CommandType.EXPAND}:
            return True
        center = cmd.target_pos or self._spawn_hatchery_pos
        if center is None:
            return False
        pos = self.decode_placement(placement_idx, center)
        x, y = int(pos.x), int(pos.y)
        # Gas buildings always target the geyser Unit — spatial cell is not the
        # build coordinate. Requiring the exact geyser tile made the policy bank
        # hundreds of minerals before a rare correct click.
        if cmd.unit_type_id in {
            UnitTypeId.EXTRACTOR,
            UnitTypeId.EXTRACTORRICH,
        }:
            if self._target_geyser_pos is not None:
                return True
            hatch = self._episode_hatch_pos()
            anchor = hatch or cmd.target_pos
            return (
                anchor is not None and self._local_geyser_unit(anchor) is not None
            )
        needs_creep = cmd.unit_type_id not in {
            UnitTypeId.HATCHERY,
            UnitTypeId.LAIR,
            UnitTypeId.HIVE,
        }
        if not self._grid_pathable(x, y):
            return False
        if needs_creep and not self._cell_has_creep(x, y):
            return False
        return True

    def _cell_has_geyser(self, x: int, y: int) -> bool:
        """True when (x, y) is the episode-local geyser cell (not any map geyser)."""
        geyser = self._target_geyser_pos
        if geyser is None:
            hatch = self._episode_hatch_pos()
            if hatch is not None:
                geyser = self._resolve_local_geyser_pos(hatch)
        if geyser is None:
            return False
        return int(geyser.x) == x and int(geyser.y) == y

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

    def _can_emit_script_step(self, step: ManagerScriptStep) -> bool:
        """Gate scripted INJECT until a queen has enough energy."""
        if step.command.command_type != CommandType.INJECT:
            return True
        return self._any_queen_can_inject()

    def _any_queen_can_inject(self) -> bool:
        queens = self._queens()
        for queen in queens:
            energy = float(getattr(queen, "energy", 0.0) or 0.0)
            if energy >= INJECT_ENERGY_COST:
                return True
        return False

    def _queens(self) -> List:
        units = getattr(self, "units", None)
        if units is None:
            return []
        try:
            return list(units(UnitTypeId.QUEEN))
        except Exception:
            return [
                u
                for u in units
                if getattr(u, "type_id", None) == UnitTypeId.QUEEN
            ]

    def _pointer_units(self, cmd: Optional[Command] = None) -> List:
        """Units selectable by the build pointer (workers, or queens for INJECT)."""
        active = cmd if cmd is not None else self.command_queue.active
        if active is not None and active.command_type == CommandType.INJECT:
            return sorted(self._queens(), key=lambda u: u.tag)
        return sorted(self.workers, key=lambda u: u.tag)

    async def on_build_substep(self):
        """Per game-step runtime: manager emit-if-idle, mining, command activation."""
        await self._suppress_seat_filler()
        self.scripted_manager.on_game_step()
        self.scripted_manager.emit_next_if_idle()
        self.command_queue.activate_next()
        self._prune_missing_worker_roles()
        self._release_idle_builders()
        if self.command_queue.standing_mine is not None:
            self._apply_standing_mine_command()
        self._apply_standing_mine_gas_command()
        self.stub_combat.apply(self, self.command_queue.standing_combat)
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
        completed_cmd = self.command_queue.active
        if self._try_complete_active_command():
            reward += float(self.rewards_cfg.get("command_complete", 15.0))
            if completed_cmd is not None and completed_cmd.command_type == CommandType.INJECT:
                reward += float(self.rewards_cfg.get("inject_reward", 20.0))
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
    def _units_per_train(unit_type: Optional[UnitTypeId]) -> int:
        """How many finished units one TRAIN order yields (Zergling egg → 2)."""
        if unit_type in {UnitTypeId.ZERGLING, UnitTypeId.ZERGLINGBURROWED}:
            return 2
        return 1

    @staticmethod
    def _train_target_count(cmd: Command) -> int:
        """How many units a TRAIN command must produce (None/<=0 => 1)."""
        raw = getattr(cmd, "count", None)
        if raw is None or int(raw) <= 0:
            return 1
        return int(raw)

    def _pending_unit_equivalent(self, unit_type: UnitTypeId) -> int:
        """Pending production counted in finished-unit equivalents."""
        try:
            pending_orders = int(self.already_pending(unit_type))
        except Exception:
            pending_orders = 0
        return pending_orders * self._units_per_train(unit_type)

    def _train_progress_units(self, unit_type: UnitTypeId) -> int:
        """Alive + pending (in unit equivalents) toward a TRAIN command."""
        return self._unit_count(unit_type) + self._pending_unit_equivalent(unit_type)

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

    def _build_macro_started(self, unit_type: UnitTypeId) -> bool:
        """True once a BUILD/EXPAND order is in flight or a local structure exists.

        Used to pipeline the next manager macro without waiting for the building
        to finish (``.ready``). Tech-gated TRAIN still waits on ``.ready``.
        """
        if self._has_structure_near_episode_hatch(unit_type, ready=False):
            return True
        try:
            return int(self.already_pending(unit_type)) > 0
        except Exception:
            return False

    def _try_complete_active_command(self) -> bool:
        """Complete the active macro command when its goal is met. Not episode done."""
        cmd = self.command_queue.active
        if cmd is None:
            return False
        completed = False
        if cmd.command_type in {CommandType.BUILD, CommandType.EXPAND}:
            # Pipeline: advance as soon as the build is started, not when finished.
            if cmd.unit_type_id is not None and self._build_macro_started(cmd.unit_type_id):
                completed = True
        elif cmd.command_type == CommandType.TRAIN:
            if cmd.unit_type_id is not None:
                target = self._train_target_count(cmd)
                # Pipeline: eggs/pending count so the next macro can start while
                # morphs finish.
                if (
                    self._train_progress_units(cmd.unit_type_id)
                    >= self._train_baseline + target
                ):
                    completed = True
        elif cmd.command_type == CommandType.INJECT:
            if self._inject_succeeded:
                completed = True
                self._inject_succeeded = False
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
        """Decode policy action into SC2 build/train/inject order."""
        self._last_build_attempt_failed = False
        cmd = self.command_queue.active
        if cmd is None or not cmd.is_macro:
            return False

        actor_idx = int(action[0])
        placement_idx = int(action[1])
        actors = self._pointer_units(cmd)
        if actor_idx >= len(actors):
            self._last_build_attempt_failed = True
            return True
        actor = actors[actor_idx]
        self._focus_worker_tag = int(actor.tag)

        if cmd.command_type == CommandType.TRAIN:
            return await self._apply_train(cmd, actor)
        if cmd.command_type == CommandType.INJECT:
            return await self._apply_inject(cmd, actor, placement_idx)

        return await self._apply_build(cmd, actor, placement_idx)

    async def _apply_build(self, cmd: Command, worker, placement_idx: int) -> bool:
        """Agent-faithful build: issue worker.build at the decoded placement cell."""
        if cmd.unit_type_id is None:
            self._last_build_attempt_failed = True
            return True

        # Already building / built near this episode's hatch — wait for complete.
        # Global ``structures(...).exists`` would treat leftover extractors/pools
        # from a prior soft-reset episode as done and skip the new build.
        if self._has_structure_near_episode_hatch(cmd.unit_type_id):
            return True
        try:
            if self.already_pending(cmd.unit_type_id) > 0:
                return True
        except Exception:
            pass

        if not self.can_afford(cmd.unit_type_id):
            # Waiting for scripted mining to bank minerals — not an invalid attempt.
            return True

        is_gas_building = cmd.unit_type_id in {
            UnitTypeId.EXTRACTOR,
            UnitTypeId.EXTRACTORRICH,
            UnitTypeId.ASSIMILATOR,
            UnitTypeId.ASSIMILATORRICH,
            UnitTypeId.REFINERY,
            UnitTypeId.REFINERYRICH,
        }
        if is_gas_building:
            # Resolve geyser from the episode hatch BEFORE placement/legality so
            # obs crop and actuation cannot stay locked on a previous episode.
            # Never fall back to a global nearest geyser (cross-base bug).
            hatch = self._episode_hatch_pos()
            anchor = hatch or self._target_geyser_pos or cmd.target_pos
            geyser = self._local_geyser_unit(anchor) if anchor is not None else None
            if geyser is None:
                self._last_build_attempt_failed = True
                return True
            cmd.target_pos = geyser.position
            self._target_geyser_pos = geyser.position

        if not self._placement_cell_legal(cmd, placement_idx):
            self._last_build_attempt_failed = True
            return True

        target = self.decode_placement(
            placement_idx, cmd.target_pos or worker.position
        )
        # Burnysc2 requires gas buildings to target the geyser Unit, not a Point2.
        if is_gas_building:
            target = geyser
        # Pull selected miner off the mining controller for this build order.
        self.worker_registry.set_role(worker.tag, WorkerRole.BUILD)
        self.do(worker.build(cmd.unit_type_id, target))
        return True

    def _train_tech_ready(self, cmd: Command) -> bool:
        """Structure prerequisites for the active TRAIN unit type."""
        unit_type = cmd.unit_type_id
        if unit_type is None:
            return False
        # Larva morphs with no building tech.
        if unit_type in {UnitTypeId.DRONE, UnitTypeId.OVERLORD}:
            return True
        # Queens train from a hatch/lair/hive; pool required in melee.
        if unit_type == UnitTypeId.QUEEN:
            halls = self.townhalls.ready if hasattr(self.townhalls, "ready") else self.townhalls
            has_hall = bool(getattr(halls, "exists", False) or len(list(halls)) > 0)
            has_pool = bool(self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists)
            return has_hall and has_pool
        if unit_type in {UnitTypeId.ZERGLING, UnitTypeId.ZERGLINGBURROWED}:
            return bool(self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists)
        if unit_type in {UnitTypeId.ROACH, UnitTypeId.ROACHBURROWED}:
            return bool(self.structures(UnitTypeId.ROACHWARREN).ready.exists)
        # Conservative default: pool for other early Zerg army units.
        return bool(self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists)

    async def _apply_train(self, cmd: Command, worker) -> bool:
        """TRAIN helper: larva morphs, or hatchery train for queens."""
        del worker
        if cmd.unit_type_id is None:
            self._last_build_attempt_failed = True
            return True
        if not self._train_tech_ready(cmd):
            self._last_build_attempt_failed = True
            return True
        target = self._train_target_count(cmd)
        goal = self._train_baseline + target
        # Zergling eggs count as 1 pending order but yield 2 units — use equivalents.
        if self._train_progress_units(cmd.unit_type_id) >= goal:
            return True
        if not self.can_afford(cmd.unit_type_id):
            return True
        if cmd.unit_type_id == UnitTypeId.QUEEN:
            halls = self.townhalls.ready if hasattr(self.townhalls, "ready") else self.townhalls
            if not getattr(halls, "exists", True) or not halls:
                return True
            hatch = halls.random if hasattr(halls, "random") else list(halls)[0]
            self.do(hatch.train(UnitTypeId.QUEEN))
            return True
        larva = self.larva
        if not larva.exists:
            return True
        self.do(larva.random.train(cmd.unit_type_id))
        return True

    def _townhall_near_cell(self, world_pos: Point2, max_dist: float = 3.5):
        """Nearest ready hatch/lair/hive to a decoded placement point."""
        halls = list(self.townhalls) if self.townhalls else []
        if not halls:
            return None
        nearest = min(halls, key=lambda h: h.distance_to(world_pos))
        if nearest.distance_to(world_pos) <= max_dist:
            return nearest
        return None

    async def _apply_inject(self, cmd: Command, queen, placement_idx: int) -> bool:
        """Agent-faithful inject: selected queen casts InjectLarva on clicked hatch."""
        from sc2.ids.ability_id import AbilityId

        if getattr(queen, "type_id", None) != UnitTypeId.QUEEN:
            self._last_build_attempt_failed = True
            return True
        energy = float(getattr(queen, "energy", 0.0) or 0.0)
        if energy < INJECT_ENERGY_COST:
            self._last_build_attempt_failed = True
            return True

        center = cmd.target_pos or queen.position
        if center is None or (
            isinstance(center, Point2) and center.x < 0 and center.y < 0
        ):
            center = (
                self.townhalls.first.position
                if self.townhalls and getattr(self.townhalls, "exists", True)
                else queen.position
            )
        world = self.decode_placement(placement_idx, center)
        hatch = self._townhall_near_cell(world)
        if hatch is None:
            self._last_build_attempt_failed = True
            return True

        ability = getattr(AbilityId, "EFFECT_INJECTLARVA", None) or getattr(
            AbilityId, "QUEENFEEDLARVA_INJECTLARVA_QUEEN", None
        )
        if ability is None:
            self._last_build_attempt_failed = True
            return True
        self.do(queen(ability, hatch))
        self._inject_succeeded = True
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
        geysers = getattr(self, "vespene_geyser", None)
        if not geysers:
            return None
        if hasattr(geysers, "exists") and not geysers.exists:
            return None
        return geysers.closest_to(pos)

    def get_obs_focus_unit_tag(self) -> int:
        return int(self._focus_worker_tag)

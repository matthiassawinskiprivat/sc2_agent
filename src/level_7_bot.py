from collections import Counter

from src.level_3_bot import Level3Bot
from src.utils.command_constants import CommandConstants
from src.sc2_bot import MapCleanupFailedError, unit_is_subject_to_map_cleanup
from src.sc2_bot_universal import SC2BotUniversal
from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    UNUSED_COMMAND_HEAD,
    build_action_head_manifest,
)
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
import random
import numpy as np
from loguru import logger
from typing import Tuple, Dict, List, Any, Optional
from scipy.optimize import linear_sum_assignment


class Level7Bot(Level3Bot):
    """
    Level 7 Bot: Combat Basics (Move vs Attack).
    Controls a specific unit type against an enemy unit type.
    """

    # Micro danger disk: enemies with weapon range above this (vs the ally) count as "ranged"
    # and get a wider danger zone (kiting / micro reward geometry).
    RANGED_ENEMY_MICRO_RANGE_CUTOFF = 2.5
    RANGED_ENEMY_DANGER_ZONE_MULT = 1.1
    @staticmethod
    def _resolve_enemies_per_target(training_conf: dict, enemy_unit_type: UnitTypeId) -> int:
        """
        How many enemy units to spawn per allied spawn slot (per target).
        Default: 2 for Zerglings (legacy curriculum), 1 for other enemy types.
        Override with top-level or env.enemies_per_target in YAML (merged into training_conf).
        """
        env_conf = training_conf.get("env", {}) if training_conf else {}
        default = 2 if enemy_unit_type == UnitTypeId.ZERGLING else 1
        raw = training_conf.get(
            "enemies_per_target",
            env_conf.get("enemies_per_target", default),
        )
        return max(1, int(raw))

    def _masking_active_unit_for_universal(self) -> Any:
        """Resolve the focal ally for universal masking (matches env focal unit)."""
        tag = getattr(self, "active_unit_tag", None)
        if tag is not None and hasattr(self, "units"):
            u = self.units.find_by_tag(tag)
            if u is not None:
                return u
        active = self.get_active_units()
        idx = int(getattr(self, "current_unit_index", 0) or 0)
        if active and 0 <= idx < len(active):
            return active[idx]
        return None

    def _any_enemy_in_tactical_fovea(self, ally: Unit) -> bool:
        """
        True if any configured enemy would be drawn in the 32×32 tactical crop around ``ally``
        (same geometry as ``ObservationTransformer.transform_tactical``).
        """
        if ally is None:
            return False
        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return False

        ux = float(ally.position.x)
        uy = float(ally.position.y)
        crop_x = ux - 16.0
        crop_y = uy - 16.0

        for e in enemies:
            if getattr(e, "is_cloaked", False) and not getattr(e, "is_detected", True):
                if not getattr(e, "is_snapshot", False):
                    continue
            dx = abs(float(e.position.x) - ux)
            dy = abs(float(e.position.y) - uy)
            if dx > 17.0 or dy > 17.0:
                continue
            px = int(e.position.x - crop_x)
            py = int(31 - (e.position.y - crop_y))
            if 0 <= px < 32 and 0 <= py < 32:
                return True
        return False

    def action_masks(self, curriculum_progress: float = 1.0) -> Any:
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
            mask = np.asarray(
                SC2BotUniversal.action_masks(self, curriculum_progress), dtype=bool
            ).copy()
            manifest = build_action_head_manifest(
                self.training_conf,
                list(
                    self.training_conf.get("action_heads")
                    or getattr(self, "action_heads", [])
                ),
            )
            active = self._masking_active_unit_for_universal()
            allow_attack = active is not None and self._any_enemy_in_tactical_fovea(active)
            if not allow_attack:
                for i in range(FIXED_UNIVERSAL_COMMAND_DIM):
                    h = manifest[i]
                    if h != UNUSED_COMMAND_HEAD and "attack" in str(h).lower():
                        mask[i] = False
            return mask
        return super().action_masks(curriculum_progress)

    async def apply_action(self, unit_tag, action):
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
            if not self.agent_system or "agent_0" not in self.agent_system.unit_agents:
                return
            unit = self.units.find_by_tag(unit_tag)
            if not unit:
                return
            # Default: no attributed target unless we issue unit.attack(enemy_unit).
            self.last_action_was_attack_by_unit[unit_tag] = False
            self.last_attack_target_tag_by_unit.pop(unit_tag, None)

            head_idx = int(action[0])
            spatial_idx = int(action[1])
            if head_idx < 0 or head_idx >= FIXED_UNIVERSAL_COMMAND_DIM:
                return
            action_heads = getattr(self, "action_heads", []) or self.training_conf.get(
                "action_heads", []
            )
            manifest = build_action_head_manifest(self.training_conf, list(action_heads))
            head_name = manifest[head_idx]
            if head_name == UNUSED_COMMAND_HEAD:
                return
            if head_idx >= len(action_heads):
                return
            agent = self.agent_system.unit_agents["agent_0"]
            world_x, world_y = agent.convert_spatial_action_to_world(head_name, spatial_idx, active_unit=unit)
            target_pos = Point2((world_x, world_y))
            self.last_target_pos = target_pos

            if "move" in head_name:
                self._combat_reward_suppressed.pop(unit_tag, None)
                self.do(unit.move(target_pos))
                return

            if "attack" in head_name:
                if not self._kite_damage_credit_ok.get(unit_tag, True):
                    self._combat_reward_suppressed[unit_tag] = True
                else:
                    self._combat_reward_suppressed[unit_tag] = False
                    self._kite_damage_credit_ok[unit_tag] = False
                self.last_action_was_attack_by_unit[unit_tag] = True
                # Issue attack at the agent-chosen world position (no assignment / nearest-enemy remap).
                self.last_attack_target_tag_by_unit[unit_tag] = None
                self.do(unit.attack(target_pos))
                return
            return
        return await super().apply_action(unit_tag, action)

    def __init__(self, conn, training_conf=None):
        super().__init__(conn, training_conf)
        self.previous_health = {} # tag -> health (enemy)
        self.previous_allied_health = {} # tag -> health (allied)
        self.previous_danger_counts = {} # tag -> enemy danger-zone count
        self.last_attack_target_tag_by_unit = {} # ally_tag -> enemy_tag (only when attacking a unit)
        self.last_action_was_attack_by_unit = {} # ally_tag -> bool
        
        # Resolve unit types from configuration
        # Look in training_conf root first, then in 'env' section
        env_conf = self.training_conf.get("env", {})
        allied_name = self.training_conf.get("unit_type", env_conf.get("unit_type", "MARINE")).upper()
        enemy_name = self.training_conf.get("enemy_type", env_conf.get("enemy_type", "ZERGLING")).upper()
        
        try:
            self.allied_unit_type = getattr(UnitTypeId, allied_name)
        except AttributeError:
            logger.warning(f"Unknown allied unit type: {allied_name}, falling back to MARINE")
            self.allied_unit_type = UnitTypeId.MARINE
            
        try:
            self.enemy_unit_type = getattr(UnitTypeId, enemy_name)
        except AttributeError:
            logger.warning(f"Unknown enemy unit type: {enemy_name}, falling back to ZERGLING")
            self.enemy_unit_type = UnitTypeId.ZERGLING

        self.enemies_per_target = self._resolve_enemies_per_target(
            self.training_conf, self.enemy_unit_type
        )

        # shoot ↔ move-away kiting chain (Level7 combat; always on)
        self._kite_damage_credit_ok: Dict[int, bool] = {}
        self._kite_phase: Dict[int, str] = {}
        self._kite_streak: Dict[int, int] = {}
        self._kiting_prev_dist: Dict[int, float] = {}
        self._kiting_chain_bonus_pending: Dict[int, float] = {}
        self._combat_reward_suppressed: Dict[int, bool] = {}

        # Filled after a successful combat reset: per-target spawn anchors, enemy tags per
        # spawn cluster, and each ally's primary target index (nearest spawn start).
        self._group_anchors: List[Point2] = []
        self._enemy_tags_by_group: List[set] = []
        self._ally_primary_group: Dict[int, int] = {}

    def _combat_topology_ready(self) -> bool:
        ga = getattr(self, "_group_anchors", None)
        eg = getattr(self, "_enemy_tags_by_group", None)
        if ga is None or eg is None:
            return False
        return (
            len(ga) == self.targets_count
            and len(eg) == self.targets_count
            and self.targets_count > 0
        )

    def _finalize_combat_spawn_topology(self, locations: List[Point2]) -> None:
        """Map allies and enemies to per-target groups using spawn layout (pair i ↔ slot i)."""
        tc = self.targets_count
        if len(locations) < tc * 2:
            self._group_anchors = []
            self._enemy_tags_by_group = []
            self._ally_primary_group = {}
            return
        ends = [locations[tc + i] for i in range(tc)]
        starts = [locations[i] for i in range(tc)]
        self._group_anchors = [Point2((float(p.x), float(p.y))) for p in ends]

        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            self._enemy_tags_by_group = [set() for _ in range(tc)]
            self._ally_primary_group = {}
            return

        groups: List[set] = [set() for _ in range(tc)]
        for e in enemies:
            gi = min(range(tc), key=lambda i: float(e.position.distance_to(ends[i])))
            groups[gi].add(e.tag)
        self._enemy_tags_by_group = groups

        self._ally_primary_group = {}
        for a in self.get_active_units():
            gi = min(range(tc), key=lambda i: float(a.position.distance_to(starts[i])))
            self._ally_primary_group[a.tag] = gi

    def _living_enemy_tags_in_group(self, group_idx: int) -> set:
        if not self._combat_topology_ready() or group_idx < 0 or group_idx >= self.targets_count:
            return set()
        alive = set()
        enemies = self.enemy_units(self.enemy_unit_type)
        if enemies.exists:
            alive = {u.tag for u in enemies}
        return alive & self._enemy_tags_by_group[group_idx]

    def _group_has_living_enemies(self, group_idx: int) -> bool:
        return len(self._living_enemy_tags_in_group(group_idx)) > 0

    def _active_target_group_for_ally(self, ally_tag: int) -> Optional[int]:
        """
        Index of the enemy pack this ally should focus on for command + constrained matching.

        Walk primary slot g0, then g0+1, … while packs still have units; once that chain
        is cleared, continue in spawn order with indices 0..g0-1 so allies can engage
        remaining groups. Without this wrap, an ally whose primary pack was cleared out
        of order (e.g. helped kill a higher-index cluster first) got no active group while
        enemies still lived — stale empty ``allowed`` sets and a missing command objective.
        """
        if not self._combat_topology_ready():
            return None
        g0 = self._ally_primary_group.get(ally_tag)
        if g0 is None:
            for g in range(self.targets_count):
                if self._group_has_living_enemies(g):
                    return g
            return None
        g0 = int(g0)
        for g in range(g0, self.targets_count):
            if self._group_has_living_enemies(g):
                return g
        for g in range(0, g0):
            if self._group_has_living_enemies(g):
                return g
        return None

    def _allowed_enemy_tags_for_ally_combat(self, ally_tag: int) -> set:
        g = self._active_target_group_for_ally(ally_tag)
        if g is None:
            return set()
        return set(self._living_enemy_tags_in_group(g))

    def _distance_to_assigned_enemy_or_none(self, unit_tag: int):
        """Distance from ally `unit_tag` to its combat-assigned enemy, or None."""
        self.refresh_combat_assignments()
        unit = self.units.find_by_tag(unit_tag)
        if unit is None:
            return None
        e_tag = self.current_assignments.get(unit_tag)
        if e_tag is None:
            return None
        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return None
        enemy = enemies.find_by_tag(e_tag)
        if enemy is None:
            return None
        return float(unit.distance_to(enemy))

    def snapshot_kiting_before_step(self, unit_tag: int) -> None:
        if unit_tag is None:
            return
        d = self._distance_to_assigned_enemy_or_none(unit_tag)
        if d is not None:
            self._kiting_prev_dist[unit_tag] = d

    def update_kiting_chain_after_step(
        self, unit_tag: int, action, prev_dist, new_dist
    ) -> None:
        """
        After physics: update kiting FSM, pending shoot→move→shoot micro bonus, and
        re-enable damage credit after a move-away step.
        """
        if unit_tag is None:
            return

        was_attack = bool(self.last_action_was_attack_by_unit.get(unit_tag, False))
        rewards_conf = self.training_conf.get("rewards", {})
        micro = float(rewards_conf.get("micro_bonus", 2.0))

        if prev_dist is not None and new_dist is not None and not was_attack:
            eps = 1e-3
            moved_away = new_dist > prev_dist + eps
            moved_closer = new_dist < prev_dist - eps
            if moved_closer:
                self._kite_streak[unit_tag] = 0
                self._kite_phase[unit_tag] = "idle"
            elif moved_away:
                self._kite_damage_credit_ok[unit_tag] = True
                if self._kite_phase.get(unit_tag, "idle") == "after_shoot_need_move":
                    self._kite_phase[unit_tag] = "after_move_need_shoot"

        if was_attack:
            phase = self._kite_phase.get(unit_tag, "idle")
            streak = int(self._kite_streak.get(unit_tag, 0))
            if phase == "after_move_need_shoot":
                bonus = (streak + 1) * micro
                self._kiting_chain_bonus_pending[unit_tag] = (
                    float(self._kiting_chain_bonus_pending.get(unit_tag, 0.0)) + bonus
                )
                self._kite_streak[unit_tag] = streak + 1
                self._kite_phase[unit_tag] = "after_shoot_need_move"
            elif phase == "after_shoot_need_move":
                self._kite_streak[unit_tag] = 0
                self._kite_phase[unit_tag] = "after_shoot_need_move"
            else:
                self._kite_phase[unit_tag] = "after_shoot_need_move"

    def _clear_kiting_state_for_tag(self, unit_tag: int) -> None:
        for d in (
            self._kite_damage_credit_ok,
            self._kite_phase,
            self._kite_streak,
            self._kiting_prev_dist,
            self._kiting_chain_bonus_pending,
            self._combat_reward_suppressed,
        ):
            d.pop(unit_tag, None)

    async def _debug_unlock_colossus_extended_thermal_lance(self):
        """
        Ensure Colossus has its range upgrades researched in training via SC2 debug commands.
        `Client.debug_upgrade()` researches upgrades in tiers; calling it twice typically unlocks
        Thermal Lance and Extended Thermal Lance for Protoss.
        """
        await self.client.debug_upgrade()
        await self.client.step()
        await self._refresh_state_local()
        await self.client.debug_upgrade()
        await self.client.step()
        await self._refresh_state_local()

    async def reset_game_state(self):
        """
        Spawns Allied and Enemy units based on configuration with high reliability.
        """
        self.previous_distances = {}
        self.current_assignments = {}
        self.previous_health = {}
        self.previous_allied_health = {}
        self.previous_danger_counts = {}
        self.last_attack_target_tag_by_unit = {}
        self.last_action_was_attack_by_unit = {}
        self._kite_damage_credit_ok.clear()
        self._kite_phase.clear()
        self._kite_streak.clear()
        self._kiting_prev_dist.clear()
        self._kiting_chain_bonus_pending.clear()
        self._combat_reward_suppressed.clear()
        self._group_anchors = []
        self._enemy_tags_by_group = []
        self._ally_primary_group = {}

        # 1. Guarantee Vision and Initial Cleanup
        # debug_show_map is a TOGGLE — calling it twice re-enables fog.
        # Only call when fog is currently enabled to avoid flipping it back on.
        if not getattr(self, "_debug_show_map_enabled", False):
            await self.client.debug_show_map()
            self._debug_show_map_enabled = True
        await self.client.step()
        await self._refresh_state_local()

        # Robust cleanup: match SC2Bot (neutral + map props like collapsible rocks excluded).
        max_cleanup_attempts = 8
        steps_after_kill = 2
        for cleanup_attempt in range(max_cleanup_attempts):
            tags_to_kill = [u.tag for u in self.all_units if unit_is_subject_to_map_cleanup(u)]
            if not tags_to_kill:
                break

            await self.client.debug_kill_unit(tags_to_kill)
            for _ in range(steps_after_kill):
                await self.client.step()
            await self._refresh_state_local()

            remaining_targets = [u for u in self.all_units if unit_is_subject_to_map_cleanup(u)]
            if not remaining_targets:
                break
            logger.warning(
                f"Cleanup attempt {cleanup_attempt+1} incomplete. "
                f"Remaining cleanup targets: {len(remaining_targets)}."
            )

        remaining = [u for u in self.all_units if unit_is_subject_to_map_cleanup(u)]
        if remaining:
            n = len(remaining)
            type_hist = Counter(getattr(u.type_id, "name", str(u.type_id)) for u in remaining)
            logger.error(
                f"Failed to clear map after cleanup loop. Remaining cleanup targets: {n}. "
                f"Types: {dict(type_hist)}. Raising MapCleanupFailedError so the env can hard-reset SC2."
            )
            raise MapCleanupFailedError(n)

        # 2. Grid Sampling for Spawns (cached — pathing grid is static during training)
        if not hasattr(self, "_cached_spawn_points") or not self._cached_spawn_points:
            grid = self.game_info.pathing_grid
            self._cached_spawn_points = [
                Point2((x, y))
                for x in range(4, grid.width - 4, 4)
                for y in range(4, grid.height - 4, 4)
                if grid[(x, y)] != 0
            ]
        valid_spawn_points = self._cached_spawn_points

        if len(valid_spawn_points) < self.targets_count * 2:
             w, h = self.game_info.map_size
             valid_spawn_points = [Point2((w/2, h/2)) for _ in range(self.targets_count * 2)]

        # 3. Spawning with Verification
        max_spawn_retries = 3
        expected_allied = self.targets_count
        expected_enemy = self.targets_count * self.enemies_per_target

        for spawn_attempt in range(max_spawn_retries):
            locations = random.sample(valid_spawn_points, self.targets_count * 2)
            
            # Spawn Pairs — batched into one RPC call to halve SC2 round-trips
            enemy_amount = self.enemies_per_target
            spawn_cmds = []
            for i in range(self.targets_count):
                start = locations[i]
                end   = locations[self.targets_count + i]
                spawn_cmds.append([self.allied_unit_type, 1,            start, self.player_id])
                spawn_cmds.append([self.enemy_unit_type,  enemy_amount, end,   2])
            await self.client.debug_create_unit(spawn_cmds)

            # Wait for engine to place units
            await self.client.step()
            await self._refresh_state_local()
            
            actual_allied = self.units(self.allied_unit_type).amount
            actual_enemy = self.enemy_units(self.enemy_unit_type).amount
            
            if actual_allied == expected_allied and actual_enemy == expected_enemy:
                logger.info(f"Spawning successful: {actual_allied} allied, {actual_enemy} enemy.")
                self._finalize_combat_spawn_topology(locations)
                if self.allied_unit_type == UnitTypeId.COLOSSUS:
                    await self._debug_unlock_colossus_extended_thermal_lance()
                return
            
            logger.warning(f"Spawning attempt {spawn_attempt+1} failed. Expected {expected_allied}/{expected_enemy}, got {actual_allied}/{actual_enemy}. Retrying cleanup.")
            
            # Cleanup and retry
            retry_tags = [u.tag for u in self.all_units if unit_is_subject_to_map_cleanup(u)]
            if retry_tags:
                await self.client.debug_kill_unit(retry_tags)
                for _ in range(steps_after_kill):
                    await self.client.step()
                await self._refresh_state_local()
            
        logger.error(f"Failed to spawn correct unit counts after {max_spawn_retries} attempts.")

    async def _refresh_state_local(self):
        """Local helper to refresh state without crashing if proto is missing."""
        obs = await self.client.observation()
        from src.utils.sc2_observation_compat import game_state_from_response_observation
        from train.envs.level_0_env import ProtoWrapper
        try:
            state = game_state_from_response_observation(obs.observation)
            self._prepare_step(state, ProtoWrapper(self.game_info._proto))
        except Exception:
            pass

    def get_active_units(self):
        """Returns active allied units."""
        if not hasattr(self, "units"): return []
        return sorted(self.units(self.allied_unit_type), key=lambda u: u.tag)

    def calculate_combat_reward(self) -> float:
        """
        Calculates reward based on damage dealt and kills.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        damage_reward = rewards_conf.get('damage_reward', 0.1)
        kill_reward = rewards_conf.get('kill_reward', 10.0)
        
        total_reward = 0.0
        
        # 1. Track Enemy Health (Damage Dealt)
        enemies = self.enemy_units(self.enemy_unit_type)
        current_enemy_tags = {u.tag for u in enemies}
        
        # Check for damage on existing enemies
        for u in enemies:
            prev = self.previous_health.get(u.tag, u.health_max + u.shield_max)
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

    def calculate_unit_centric_combat_delta(self, acting_unit_tag: int) -> Tuple[float, float]:
        """
        Returns (attributed_reward_for_acting_unit, pooled_reward_delta).

        Attribution is conservative / "100% certain":
        - Only attribute damage/kill when the acting unit issued `attack` on a specific enemy Unit
          (`last_action_was_attack_by_unit[tag]` and a non-None `last_attack_target_tag_by_unit[tag]`),
          and the step's enemy damage/kill deltas are uniquely attributable to that target.
        - Everything else is pooled.
        """
        rewards_conf = self.training_conf.get("rewards", {})
        damage_reward = float(rewards_conf.get("damage_reward", 0.1))
        kill_reward = float(rewards_conf.get("kill_reward", 10.0))

        enemies = self.enemy_units(self.enemy_unit_type)
        enemy_list = list(enemies) if enemies.exists else []
        current_enemy_tags = {u.tag for u in enemy_list}

        # Compute per-enemy damage deltas.
        damage_events: Dict[int, float] = {}
        for u in enemy_list:
            curr = float(u.health + u.shield)
            if u.tag not in self.previous_health:
                # Initialize baseline on first sight; don't award/penalize.
                self.previous_health[u.tag] = curr
                continue
            prev = float(self.previous_health.get(u.tag, curr))
            dmg = max(0.0, prev - curr)
            if dmg > 0.0:
                damage_events[u.tag] = dmg
            self.previous_health[u.tag] = curr

        # Determine deaths since last step (based on tracked tags).
        prev_tags = set(self.previous_health.keys())
        died_tags = prev_tags - current_enemy_tags

        # Convert to scalar rewards (damage, kill) by tag.
        damage_rewards_by_tag = {t: d * damage_reward for t, d in damage_events.items()}
        kill_rewards_by_tag = {t: kill_reward for t in died_tags}

        # Clean up previous_health for dead enemies.
        for t in died_tags:
            self.previous_health.pop(t, None)

        # Attribution decision.
        target_tag = None
        if acting_unit_tag is not None and self.last_action_was_attack_by_unit.get(acting_unit_tag, False):
            target_tag = self.last_attack_target_tag_by_unit.get(acting_unit_tag)

        attributed = 0.0
        pooled = 0.0

        # Damage: attribute only if exactly one damaged enemy and it matches target.
        if target_tag is not None and len(damage_rewards_by_tag) == 1 and target_tag in damage_rewards_by_tag:
            attributed += damage_rewards_by_tag[target_tag]
        else:
            pooled += sum(damage_rewards_by_tag.values())

        # Kills: attribute only if exactly one death and it matches target.
        if target_tag is not None and len(kill_rewards_by_tag) == 1 and target_tag in kill_rewards_by_tag:
            attributed += kill_rewards_by_tag[target_tag]
        else:
            pooled += sum(kill_rewards_by_tag.values())

        if (
            acting_unit_tag is not None
            and self._combat_reward_suppressed.pop(acting_unit_tag, False)
        ):
            return 0.0, 0.0

        return attributed, pooled

    def calculate_allied_attrition_penalty(self) -> float:
        """
        Calculates penalty based on allied damage taken and units killed.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        damage_penalty = rewards_conf.get('damage_penalty', -0.1)
        kill_penalty = rewards_conf.get('kill_penalty', -10.0)
        
        total_penalty = 0.0
        
        allies = self.units(self.allied_unit_type)
        current_allied_tags = {u.tag for u in allies}
        
        # 1. Track Allied Damage
        for u in allies:
            prev = self.previous_allied_health.get(u.tag, u.health_max + u.shield_max)
            curr = u.health + u.shield
            damage = max(0, prev - curr)
            if damage > 0:
                total_penalty += damage * damage_penalty
            self.previous_allied_health[u.tag] = curr
            
        # 2. Track Allied Kills (Deaths)
        tracked_tags = set(self.previous_allied_health.keys())
        died_tags = tracked_tags - current_allied_tags
        
        for tag in died_tags:
            total_penalty += kill_penalty
            del self.previous_allied_health[tag]
            
        return total_penalty

    def calculate_unattributed_allied_attrition_delta(self) -> float:
        """
        Unattributed allied attrition delta (damage taken, deaths). This is treated as pooled
        reward in unit-centric env mode because it cannot be reliably assigned to one acting unit.
        """
        return self.calculate_allied_attrition_penalty()

    def get_closest_targets_combat(self) -> Dict[int, int]:
        """
        Returns current optimal combat assignments (Hungarian).
        """
        self.refresh_combat_assignments()
        return dict(self.current_assignments)

    def get_optimal_assignments_combat(self) -> Dict[int, int]:
        """
        Hungarian assignment (same strategy as Level3): minimize total ally-enemy distance.
        """
        allies = self.get_active_units()
        enemies = self.enemy_units(self.enemy_unit_type)
        if not allies or not enemies.exists:
            return {}

        enemy_list = list(enemies)
        if not enemy_list:
            return {}

        num_allies = len(allies)
        num_enemies = len(enemy_list)
        dist_matrix = np.zeros((num_allies, num_enemies), dtype=np.float32)

        for i, ally in enumerate(allies):
            for j, enemy in enumerate(enemy_list):
                dist_matrix[i, j] = ally.distance_to(enemy)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        assignment = {}
        for r, c in zip(row_ind, col_ind):
            assignment[allies[r].tag] = enemy_list[c].tag

        # Surplus allies: assign remaining allies to their closest enemy.
        if num_allies > num_enemies:
            for ally in allies:
                if ally.tag not in assignment:
                    closest_enemy = min(enemy_list, key=lambda enemy: ally.distance_to(enemy))
                    assignment[ally.tag] = closest_enemy.tag

        return assignment

    def refresh_combat_assignments(self) -> Dict[int, int]:
        """
        Ensures every active allied unit has a live enemy target whenever enemies exist.
        With spawn topology (after a successful Level7 reset), each ally is matched to the
        enemy pack for its target slot; otherwise the legacy global Hungarian matcher is used.
        """
        if self._combat_topology_ready():
            return self._refresh_combat_assignments_grouped()
        return self._refresh_combat_assignments_global()

    def _refresh_combat_assignments_global(self) -> Dict[int, int]:
        """
        Recomputes assignments with sticky Hungarian matching after state changes.
        1) Keep previous valid pairs (consistency)
        2) Hungarian-match remaining allies to remaining enemies (one-to-one)
        3) Assign leftover allies to nearest enemy (many-to-one fallback)
        """
        allies = self.get_active_units()
        enemies = self.enemy_units(self.enemy_unit_type)

        if not allies or not enemies.exists:
            self.current_assignments = {}
            return self.current_assignments

        enemy_list = list(enemies)
        if not enemy_list:
            self.current_assignments = {}
            return self.current_assignments

        ally_by_tag = {ally.tag: ally for ally in allies}
        enemy_by_tag = {enemy.tag: enemy for enemy in enemy_list}

        assignment = {}

        # 1) Preserve existing valid assignments for stability.
        prev_assignments = dict(getattr(self, "current_assignments", {}) or {})
        for ally_tag, enemy_tag in prev_assignments.items():
            if ally_tag in ally_by_tag and enemy_tag in enemy_by_tag and ally_tag not in assignment:
                assignment[ally_tag] = enemy_tag

        # 2) Hungarian for remaining unassigned allies/enemies.
        remaining_allies = [ally for ally in allies if ally.tag not in assignment]
        assigned_enemy_tags = set(assignment.values())
        remaining_enemies = [enemy for enemy in enemy_list if enemy.tag not in assigned_enemy_tags]

        if remaining_allies and remaining_enemies:
            num_allies = len(remaining_allies)
            num_enemies = len(remaining_enemies)
            dist_matrix = np.zeros((num_allies, num_enemies), dtype=np.float32)
            for i, ally in enumerate(remaining_allies):
                for j, enemy in enumerate(remaining_enemies):
                    dist_matrix[i, j] = ally.distance_to(enemy)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            for r, c in zip(row_ind, col_ind):
                assignment[remaining_allies[r].tag] = remaining_enemies[c].tag

        # Fill any missing ally tags defensively to guarantee full coverage.
        for ally in allies:
            if ally.tag not in assignment:
                fallback_enemy = min(enemy_list, key=lambda enemy: ally.distance_to(enemy))
                assignment[ally.tag] = fallback_enemy.tag

        self.current_assignments = assignment
        return self.current_assignments

    def _refresh_combat_assignments_grouped(self) -> Dict[int, int]:
        """Sticky Hungarian matching restricted to each ally's current enemy pack."""
        _ILLEGAL = 1.0e9
        allies = self.get_active_units()
        enemies = self.enemy_units(self.enemy_unit_type)

        if not allies or not enemies.exists:
            self.current_assignments = {}
            return self.current_assignments

        enemy_list = list(enemies)
        if not enemy_list:
            self.current_assignments = {}
            return self.current_assignments

        ally_by_tag = {ally.tag: ally for ally in allies}
        enemy_by_tag = {enemy.tag: enemy for enemy in enemy_list}

        allowed: Dict[int, set] = {
            a.tag: self._allowed_enemy_tags_for_ally_combat(a.tag) for a in allies
        }

        assignment: Dict[int, int] = {}
        prev_assignments = dict(getattr(self, "current_assignments", {}) or {})

        for ally_tag, enemy_tag in prev_assignments.items():
            if ally_tag not in ally_by_tag or enemy_tag not in enemy_by_tag:
                continue
            if ally_tag in assignment:
                continue
            if enemy_tag in allowed.get(ally_tag, set()):
                assignment[ally_tag] = enemy_tag

        remaining_allies = [ally for ally in allies if ally.tag not in assignment]
        assigned_enemy_tags = set(assignment.values())
        remaining_enemies = [e for e in enemy_list if e.tag not in assigned_enemy_tags]

        if remaining_allies and remaining_enemies:
            na = len(remaining_allies)
            ne = len(remaining_enemies)
            dist_matrix = np.full((na, ne), _ILLEGAL, dtype=np.float32)
            for i, ally in enumerate(remaining_allies):
                allow = allowed.get(ally.tag, set())
                for j, enemy in enumerate(remaining_enemies):
                    if enemy.tag in allow:
                        dist_matrix[i, j] = ally.distance_to(enemy)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            for r, c in zip(row_ind, col_ind):
                if float(dist_matrix[r, c]) < _ILLEGAL * 0.5:
                    assignment[remaining_allies[r].tag] = remaining_enemies[c].tag

        for ally in allies:
            if ally.tag in assignment:
                continue
            allow = allowed.get(ally.tag, set())
            pool = [e for e in enemy_list if e.tag in allow]
            if pool:
                best = min(pool, key=lambda e: ally.distance_to(e))
                assignment[ally.tag] = best.tag
            elif enemy_list:
                best = min(enemy_list, key=lambda e: ally.distance_to(e))
                assignment[ally.tag] = best.tag

        self.current_assignments = assignment
        return self.current_assignments

    def _get_enemy_target_for_unit(self, unit: Unit) -> Unit:
        """
        Returns a live enemy target for `unit` whenever any enemy exists.
        """
        if unit is None:
            return None

        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return None

        assignment = self.refresh_combat_assignments()
        assigned_tag = assignment.get(unit.tag)
        if assigned_tag is not None:
            assigned_enemy = enemies.find_by_tag(assigned_tag)
            if assigned_enemy is not None:
                return assigned_enemy

        # Fallback: nearest currently visible enemy so command_target is never lost.
        enemy_list = list(enemies)
        if not enemy_list:
            return None
        return min(enemy_list, key=lambda enemy: unit.distance_to(enemy))

    def get_assigned_distances_combat(self) -> Dict[int, float]:
        """
        Calculates greedy assignment and returns distances to closest enemy units.
        """
        self.refresh_combat_assignments()
        distances = {}
        enemies = self.enemy_units(self.enemy_unit_type)
        for u_tag, e_tag in self.current_assignments.items():
            unit = self.units.find_by_tag(u_tag)
            enemy = enemies.find_by_tag(e_tag) if enemies.exists else None
            if unit and enemy:
                distances[u_tag] = unit.distance_to(enemy)
        return distances

    @staticmethod
    def _weapon_range_attacker_vs_defender(attacker, defender) -> float:
        """Weapon range from attacker to a specific defender (ground vs air-aware)."""
        if attacker is None or defender is None:
            return 0.0
        if bool(getattr(defender, "is_flying", False)):
            air_r = getattr(attacker, "air_range", None)
            if air_r is not None and float(air_r) > 0.0:
                return float(air_r)
        gr = getattr(attacker, "ground_range", None)
        if gr is not None:
            return float(gr)
        return 1.0

    def _is_in_enemy_danger_zone(self, ally, enemy) -> bool:
        """
        Whether `ally` is currently in `enemy`'s effective danger zone.
        Danger zone radius is max(enemy_range_vs_ally, 0.5 * ally_range_vs_enemy).
        For ranged enemies (see `RANGED_ENEMY_MICRO_RANGE_CUTOFF`), that radius is multiplied
        by `RANGED_ENEMY_DANGER_ZONE_MULT` (default +10%).
        """
        if ally is None or enemy is None:
            return False
        eps = 1e-4
        dist = ally.distance_to(enemy)
        ally_range = self._weapon_range_attacker_vs_defender(ally, enemy)
        enemy_range = self._weapon_range_attacker_vs_defender(enemy, ally)
        threat_radius = max(enemy_range, 0.5 * ally_range)
        if enemy_range > float(self.RANGED_ENEMY_MICRO_RANGE_CUTOFF):
            threat_radius *= float(self.RANGED_ENEMY_DANGER_ZONE_MULT)
        return dist <= threat_radius + eps

    def _count_enemy_danger_zones_for_unit(self, unit: Unit) -> int:
        """Counts how many enemy units currently include `unit` in their danger zone."""
        if unit is None:
            return 0
        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return 0
        count = 0
        for enemy in enemies:
            if self._is_in_enemy_danger_zone(unit, enemy):
                count += 1
        return count

    def _count_kiting_sweet_spots_for_unit(self, unit: Unit) -> int:
        """
        Counts enemies where `unit` can shoot (within unit range) while staying outside that
        enemy's danger zone. This is the kiting \"sweet spot\".
        """
        if unit is None:
            return 0
        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return 0
        eps = 1e-4
        count = 0
        for enemy in enemies:
            if enemy is None:
                continue
            dist = unit.distance_to(enemy)
            unit_range = self._weapon_range_attacker_vs_defender(unit, enemy)
            if dist <= unit_range + eps and not self._is_in_enemy_danger_zone(unit, enemy):
                count += 1
        return count

    def calculate_micro_bonus(self, unit_tag: int) -> float:
        """
        Kiting reward channel:
        - Sweet spot (can shoot while outside danger): +micro_bonus * sweet_spot_count
        - Inside danger this step: -micro_bonus * current_danger_count
        - Exited danger (prev>0 and now==0): +1.5 * micro_bonus * prev_danger_count
        """
        if unit_tag is None:
            return 0.0

        unit = self.units.find_by_tag(unit_tag)
        if unit is None:
            self.previous_danger_counts.pop(unit_tag, None)
            self._clear_kiting_state_for_tag(unit_tag)
            return 0.0

        rewards_conf = self.training_conf.get("rewards", {})
        micro_bonus = float(rewards_conf.get("micro_bonus", 2.0))
        if micro_bonus == 0.0:
            self.previous_danger_counts[unit_tag] = self._count_enemy_danger_zones_for_unit(unit)
            return float(self._kiting_chain_bonus_pending.pop(unit_tag, 0.0))

        prev_count = int(self.previous_danger_counts.get(unit_tag, 0))
        current_count = int(self._count_enemy_danger_zones_for_unit(unit))
        sweet_spot_count = int(self._count_kiting_sweet_spots_for_unit(unit))

        reward = 0.0
        if current_count == 0 and sweet_spot_count > 0:
            reward += micro_bonus * sweet_spot_count
        if current_count > 0:
            reward -= micro_bonus * current_count
        elif prev_count > 0:
            reward += 1.5 * micro_bonus * prev_count

        self.previous_danger_counts[unit_tag] = current_count
        reward += float(self._kiting_chain_bonus_pending.pop(unit_tag, 0.0))
        return reward

    def _compute_distance_reward_for_assigned_pair(self, unit_tag: int, dist_multiplier: float) -> float:
        """
        Distance shaping for one ally and its assigned enemy.
        Shaping is (prev_dist - dist) * dist_multiplier only while the ally is NOT in
        range to attack the assigned enemy; otherwise shaping is 0.
        """
        unit = self.units.find_by_tag(unit_tag)
        if unit is None:
            self.previous_distances.pop(unit_tag, None)
            return 0.0

        e_tag = self.current_assignments.get(unit_tag)
        enemies = self.enemy_units(self.enemy_unit_type)
        enemy = enemies.find_by_tag(e_tag) if e_tag is not None and enemies.exists else None
        if enemy is None:
            self.previous_distances.pop(unit_tag, None)
            return 0.0

        dist = unit.distance_to(enemy)
        ally_range = self._weapon_range_attacker_vs_defender(unit, enemy)
        eps = 1e-4

        shaping = 0.0
        if dist > ally_range + eps:
            prev_dist = self.previous_distances.get(unit_tag)
            if prev_dist is not None:
                shaping = (prev_dist - dist) * dist_multiplier

        self.previous_distances[unit_tag] = dist
        return shaping

    def queue_idle_combat_orders(self, skip_unit_tag: int = None) -> int:
        """
        Keep assigned allies engaged between their policy turns.
        Queues attack orders only for units that are currently idle.
        """
        assignments = self.refresh_combat_assignments()
        if not assignments:
            return 0

        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return 0

        queued = 0
        for ally in self.get_active_units():
            if skip_unit_tag is not None and ally.tag == skip_unit_tag:
                continue
            if ally.tag not in assignments:
                continue

            # Do not override ongoing unit orders; just fill true idles.
            if getattr(ally, "orders", None):
                continue

            enemy = enemies.find_by_tag(assignments[ally.tag])
            if enemy is None:
                continue
            self.do(ally.attack(enemy))
            queued += 1
        return queued


    def calculate_distance_reward(self, unit_tag: int = None) -> float:
        """
        Distance shaping toward the assigned enemy with range-aware gating.

        - While the ally can attack the assigned enemy (within ally weapon range), approach
          shaping is zero.
        """
        rewards_conf = self.training_conf.get('rewards', {})
        dist_multiplier = rewards_conf.get('distance_multiplier', 0.1)

        self.refresh_combat_assignments()

        if unit_tag is not None:
            if unit_tag not in self.current_assignments:
                self.previous_distances.pop(unit_tag, None)
                return 0.0
            return self._compute_distance_reward_for_assigned_pair(unit_tag, dist_multiplier)

        total = 0.0
        for u_tag in list(self.current_assignments.keys()):
            total += self._compute_distance_reward_for_assigned_pair(u_tag, dist_multiplier)
        return total

    def get_observation_for_unit(self, unit_tag) -> Tuple[Any, Dict]:
        unit = self.units.find_by_tag(unit_tag)
        if not unit:
            return self._get_zero_observation(), {}

        original_unit = self.zergling
        self.zergling = unit

        if self._combat_topology_ready():
            g = self._active_target_group_for_ally(unit.tag)
            target_pos = self._group_anchors[g] if g is not None else None
        else:
            target = self._get_enemy_target_for_unit(unit)
            target_pos = target.position if target else None

        arch = self.training_conf.get("agent_architecture", "original")
        high_cmd = self.training_conf.get("high_level_command", 0)
        # Universal micro: a fixed "attack" directive makes command_vector[ATTACK]=1 every
        # step, which strongly correlates with the attack action head. Use a neutral scout
        # slot; objective location still comes from enemy layers and command_target.
        if arch == "universal":
            high_cmd = CommandConstants.SCOUT

        raw_obs = {
            'units': self.units,
            'selected_units': [unit],
            'enemy_units': self.enemy_units,
            'enemy_structures': self.enemy_structures,
            'structures': self.structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': getattr(self, "game_info", getattr(self, "_game_info", None)),
            'game_data': getattr(self, "game_data", getattr(self, "_game_data", None)),
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos,
            'high_level_command': high_cmd,
        }
        
        if self.agent_system:
             processed_obs = self.agent_system.process_observation(
                 observation=raw_obs, active_unit=unit
             )["agent_0"]
        else:
             processed_obs = self._get_zero_observation()
             
        self.zergling = original_unit
        return processed_obs, {}

    def _get_zero_observation(self):
        """Returns a zero observation matching the expected observation structure."""
        arch = self.training_conf.get("agent_architecture", "original")
        active_layers = self.training_conf.get("active_layers", [])
        n_channels = len(active_layers) if active_layers else 5
        if arch == "universal":
            active_tactical = self.training_conf.get("active_tactical_layers", active_layers)
            n_tactical = len(active_tactical) if active_tactical else n_channels
            return {
                "map": np.zeros((n_channels, 128, 128), dtype=np.float32),
                "tactical": np.zeros((n_tactical, 32, 32), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32),
            }
        return np.zeros((n_channels, 128, 128), dtype=np.float32)

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
from src.utils.unit_feature_helpers import (
    disable_cyclone_lock_on_autocast,
    issue_aoe_at_point,
    issue_attack_at_point,
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

    # Must match ``ObservationTransformer.transform_tactical`` allied/enemy draw culling
    # (|dx| > HALF or |dy| > HALF → outside the 32×32 fovea centered on the active unit).
    _TACTICAL_FOVEA_MAX_DELTA: float = 17.0
    _USE_GLOBAL_NO_ENEMY_RADIUS: float = 20.0

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

    def action_masks(self, curriculum_progress: float = 1.0) -> Any:
        arch = self.training_conf.get("agent_architecture", "original")
        if arch == "universal":
            return SC2BotUniversal.action_masks(self, curriculum_progress)
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
            self.last_action_was_move_by_unit[unit_tag] = False
            self.last_move_head_by_unit[unit_tag] = None
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
                self.last_action_was_move_by_unit[unit_tag] = True
                self.last_move_head_by_unit[unit_tag] = head_name
                self._attack_order_weapon_cd_before.pop(unit_tag, None)
                self.do(unit.move(target_pos))
                return

            if "attack" in head_name:
                self.last_action_was_attack_by_unit[unit_tag] = True
                self._attack_order_weapon_cd_before[unit_tag] = float(
                    getattr(unit, "weapon_cooldown", 0.0) or 0.0
                )
                # Hard-target enemy in the clicked cell when present; else attack-move.
                # Swarm Hosts map attack → Spawn Locusts; Corruptors spray buildings.
                hard_target = issue_attack_at_point(self, unit, target_pos)
                if hard_target is not None:
                    self.last_attack_target_tag_by_unit[unit_tag] = hard_target.tag
                    return
                # Attribution bookkeeping when attack-moving: nearest enemy to the click.
                enemies = self.enemy_units(self.enemy_unit_type)
                if enemies is not None and getattr(enemies, "exists", False):
                    nearest_enemy = min(
                        enemies,
                        key=lambda enemy: float(enemy.position.distance_to(target_pos)),
                    )
                    self.last_attack_target_tag_by_unit[unit_tag] = nearest_enemy.tag
                else:
                    self.last_attack_target_tag_by_unit[unit_tag] = None
                return

            if "aoe" in head_name:
                issue_aoe_at_point(self, unit, target_pos)
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
        self.last_action_was_move_by_unit: Dict[int, bool] = {}  # universal: last policy step was move
        self.last_move_head_by_unit: Dict[int, Optional[str]] = {}  # universal: move head name used
        self._pre_action_enemy_within_use_global_radius: Dict[int, bool] = {}
        
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

        self._allied_took_damage_this_step: Dict[int, bool] = {}
        # Kiting shaping state — all three dicts are populated by
        # record_pre_action_kiting_state() BEFORE the action is applied each step.
        # Storing the step-start (pre-action) baseline makes the retreat-reward
        # comparison single-step accurate even in multi-unit cycling, where the
        # previous active-turn baseline would span N game steps and be polluted by
        # enemy approach between turns.
        self._kiting_simple_prev_dist: Dict[int, float] = {}          # dist at step start
        self._prev_enemy_in_attack_range: Dict[int, bool] = {}         # enemy in contact range at step start
        self._micro_under_fire_min_dist_prev: Dict[int, Optional[float]] = {}  # min shooting-enemy dist at step start
        self._attack_order_weapon_cd_before: Dict[int, float] = {}
        self._pre_action_weapon_cd_by_unit: Dict[int, float] = {}
        # Per-unit combat reward buffer: damage/kill credit accumulates here and is flushed
        # the next time that unit becomes the acting unit. Even without a bot-side auto-attack
        # helper, SC2's default stance lets a unit auto-fire at enemies in range while its
        # previous order is winding down (e.g. attack-move completing on a dead target), so
        # damage can still land between this unit's consecutive policy turns.
        self._pending_rewards: Dict[int, float] = {}
        # Last observed position per enemy tag — used to attribute kills to closest ally.
        self._enemy_last_known_pos: Dict[int, Any] = {}
        # Last assigned enemy tag seen during kiting shaping — used to detect assignment
        # changes so stale per-enemy kiting state can be invalidated promptly.
        self._prev_kiting_enemy_tag: Dict[int, Optional[int]] = {}

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

    def _clear_kiting_state_for_tag(self, unit_tag: int) -> None:
        for d in (
            self._micro_under_fire_min_dist_prev,
            self._allied_took_damage_this_step,
            self._kiting_simple_prev_dist,
            self._prev_enemy_in_attack_range,
            self._attack_order_weapon_cd_before,
            self._pre_action_weapon_cd_by_unit,
            self._prev_kiting_enemy_tag,
        ):
            d.pop(unit_tag, None)
        self.last_action_was_move_by_unit.pop(unit_tag, None)
        self.last_move_head_by_unit.pop(unit_tag, None)
        self._pre_action_enemy_within_use_global_radius.pop(unit_tag, None)

    def _any_enemy_within_radius(self, unit: Unit, radius: float) -> bool:
        enemies = self.enemy_units(self.enemy_unit_type)
        if enemies is None or not getattr(enemies, "exists", False):
            return False
        eps = 1e-3
        for enemy in enemies:
            if float(unit.distance_to(enemy)) <= radius + eps:
                return True
        return False

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
        self.last_action_was_move_by_unit.clear()
        self.last_move_head_by_unit.clear()
        self._pre_action_enemy_within_use_global_radius.clear()
        self._micro_under_fire_min_dist_prev.clear()
        self._allied_took_damage_this_step.clear()
        self._kiting_simple_prev_dist.clear()
        self._prev_enemy_in_attack_range.clear()
        self._attack_order_weapon_cd_before.clear()
        self._pre_action_weapon_cd_by_unit.clear()
        self._pending_rewards.clear()
        self._enemy_last_known_pos.clear()
        self._prev_kiting_enemy_tag.clear()
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
                await disable_cyclone_lock_on_autocast(self)
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

    def _closest_active_ally_tag(self, reference_pos, allies: list) -> Optional[int]:
        """Tag of the active ally geometrically closest to reference_pos. None if allies is empty."""
        if not allies:
            return None
        return min(allies, key=lambda a: float(a.distance_to(reference_pos))).tag

    def calculate_unit_centric_combat_delta(self, acting_unit_tag: int) -> Tuple[float, float]:
        """
        Attributes damage and kills to the most plausible attacker, buffers pending rewards per
        allied unit, and flushes the acting unit's accumulated buffer.

        Attribution rules
        -----------------
        - Damage on the enemy the acting unit **directly targeted** (nearest enemy to the
          policy-chosen attack point, stored in ``last_attack_target_tag_by_unit``) is
          attributed to the acting unit — the causal link is unambiguous.
        - Damage on **any other enemy** is attributed to the closest active ally to that
          enemy, mirroring SC2's auto-attack resolution.
        - Kills follow the same split: direct-target kill → acting unit; other kills →
          closest active ally to the enemy's last known position.

        Cross-step buffering
        --------------------
        Rewards accumulate in ``_pending_rewards[ally_tag]`` and are not consumed until
        that ally becomes the acting unit. This correctly attributes rewards earned by
        SC2's default-stance auto-fire (a unit shooting an in-range enemy while its prior
        order is still resolving) that lands during a different unit's policy step.

        Return value: ``(acting_unit_attributed_reward, 0.0)`` — the second element is
        always zero and is kept only for call-site compatibility.
        """
        rewards_conf = self.training_conf.get("rewards", {})
        damage_reward = float(rewards_conf.get("damage_reward", 0.1))
        kill_reward = float(rewards_conf.get("kill_reward", 10.0))

        allies = self.get_active_units()
        enemies = self.enemy_units(self.enemy_unit_type)
        enemy_list = list(enemies) if enemies.exists else []
        current_enemy_tags = {u.tag for u in enemy_list}

        # The enemy the acting unit explicitly targeted this step (None if move action).
        direct_target = None
        if self.last_action_was_attack_by_unit.get(acting_unit_tag, False):
            direct_target = self.last_attack_target_tag_by_unit.get(acting_unit_tag)

        for u in enemy_list:
            curr = float(u.health + u.shield)
            # Track position for kill attribution even if no damage this step.
            self._enemy_last_known_pos[u.tag] = u.position
            if u.tag not in self.previous_health:
                self.previous_health[u.tag] = curr
                continue
            prev = float(self.previous_health[u.tag])
            dmg = max(0.0, prev - curr)
            if dmg > 0.0:
                if u.tag == direct_target:
                    attacker_tag = acting_unit_tag
                else:
                    attacker_tag = (
                        self._closest_active_ally_tag(u.position, allies)
                        or acting_unit_tag
                    )
                self._pending_rewards[attacker_tag] = (
                    self._pending_rewards.get(attacker_tag, 0.0) + dmg * damage_reward
                )
            self.previous_health[u.tag] = curr

        prev_tags = set(self.previous_health.keys())
        died_tags = prev_tags - current_enemy_tags
        for t in died_tags:
            if t == direct_target:
                attacker_tag = acting_unit_tag
            else:
                last_pos = self._enemy_last_known_pos.get(t)
                attacker_tag = (
                    (self._closest_active_ally_tag(last_pos, allies) if last_pos is not None else None)
                    or acting_unit_tag
                )
            self._pending_rewards[attacker_tag] = (
                self._pending_rewards.get(attacker_tag, 0.0) + kill_reward
            )
            self.previous_health.pop(t, None)
            self._enemy_last_known_pos.pop(t, None)

        # Flush this unit's accumulated buffer into the step reward.
        attributed = self._pending_rewards.pop(acting_unit_tag, 0.0)
        return attributed, 0.0

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
        self._allied_took_damage_this_step = {}
        for u in allies:
            prev = self.previous_allied_health.get(u.tag, u.health_max + u.shield_max)
            curr = u.health + u.shield
            damage = max(0, prev - curr)
            self._allied_took_damage_this_step[u.tag] = damage > 0
            if damage > 0:
                total_penalty += damage * damage_penalty
            self.previous_allied_health[u.tag] = curr
            
        # 2. Track Allied Kills (Deaths)
        tracked_tags = set(self.previous_allied_health.keys())
        died_tags = tracked_tags - current_allied_tags
        
        for tag in died_tags:
            self._allied_took_damage_this_step[tag] = True
            total_penalty += kill_penalty
            del self.previous_allied_health[tag]
            
        return total_penalty

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
    def _get_unit_radius(unit) -> float:
        """Collision radius of ``unit``; defaults to 0.5 when not available."""
        try:
            return float(unit.radius)
        except (TypeError, AttributeError, ValueError):
            return 0.5

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
        """
        if ally is None or enemy is None:
            return False
        eps = 1e-4
        dist = ally.distance_to(enemy)
        ally_range = self._weapon_range_attacker_vs_defender(ally, enemy)
        enemy_range = self._weapon_range_attacker_vs_defender(enemy, ally)
        threat_radius = max(enemy_range, 0.5 * ally_range)
        return dist <= threat_radius + eps

    def _is_enemy_in_tactical_fovea(self, focal: Unit, enemy: Unit) -> bool:
        """True if ``enemy`` would be drawn in the active unit's 32×32 tactical obs (not off-map)."""
        if focal is None or enemy is None:
            return False
        h = float(self._TACTICAL_FOVEA_MAX_DELTA)
        dx = abs(float(enemy.position.x) - float(focal.position.x))
        dy = abs(float(enemy.position.y) - float(focal.position.y))
        return dx <= h and dy <= h

    def _min_dist_to_shooting_enemies_in_fovea(self, ally: Unit) -> Optional[float]:
        """
        Minimum distance from ``ally`` to an enemy in the tactical fovea that can shoot ``ally``
        (ally within that enemy's weapon range). None if no such enemy.
        """
        if ally is None:
            return None
        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return None
        eps = 1e-4
        best: Optional[float] = None
        for enemy in enemies:
            if enemy is None:
                continue
            if not self._is_enemy_in_tactical_fovea(ally, enemy):
                continue
            dist = float(ally.distance_to(enemy))
            enemy_range = self._weapon_range_attacker_vs_defender(enemy, ally)
            # SC2 weapon data gives the nominal range; the engine adds unit collision
            # radii before triggering attacks.  For melee units (e.g. Zergling range 0.1)
            # the raw value is smaller than the physical minimum contact distance, so we
            # use max(nominal_range, sum-of-radii) to correctly detect "enemy is shooting".
            effective_range = max(enemy_range, self._get_unit_radius(enemy) + self._get_unit_radius(ally))
            if dist <= effective_range + eps:
                best = dist if best is None else min(best, dist)
        return best

    def _count_enemy_danger_zones_for_unit(self, unit: Unit) -> int:
        """
        Counts enemies in the tactical fovea whose danger zone currently contains ``unit``.
        Enemies outside the 32×32-style crop around ``unit`` are ignored (matches tactical obs).
        """
        if unit is None:
            return 0
        enemies = self.enemy_units(self.enemy_unit_type)
        if not enemies.exists:
            return 0
        count = 0
        for enemy in enemies:
            if not self._is_enemy_in_tactical_fovea(unit, enemy):
                continue
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

    def record_pre_action_kiting_state(self, unit_tag: int) -> None:
        """
        Snapshot kiting-relevant state BEFORE the action is applied and the game steps.

        Must be called from _async_step immediately before apply_action so that
        calculate_kiting_simple_shaping can compare the post-step state against the
        state the agent actually observed — a single-frame baseline.  Without this,
        the retreat-reward comparison spans N game steps (full unit-cycling round),
        causing the Zergling approach between turns to shrink _kiting_simple_prev_dist
        below the post-step distance even when the agent correctly issued a move.
        """
        unit = self.units.find_by_tag(unit_tag)
        if unit is None:
            self._kiting_simple_prev_dist.pop(unit_tag, None)
            self._prev_enemy_in_attack_range.pop(unit_tag, None)
            self._micro_under_fire_min_dist_prev.pop(unit_tag, None)
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            self._prev_kiting_enemy_tag.pop(unit_tag, None)
            self._pre_action_enemy_within_use_global_radius.pop(unit_tag, None)
            return

        self._pre_action_weapon_cd_by_unit[unit_tag] = float(
            getattr(unit, "weapon_cooldown", 0.0) or 0.0
        )
        self._pre_action_enemy_within_use_global_radius[unit_tag] = self._any_enemy_within_radius(
            unit, self._USE_GLOBAL_NO_ENEMY_RADIUS
        )

        self.refresh_combat_assignments()
        e_tag = self.current_assignments.get(unit_tag)
        enemies = self.enemy_units(self.enemy_unit_type)
        enemy = enemies.find_by_tag(e_tag) if e_tag is not None and enemies.exists else None
        if enemy is None:
            self._kiting_simple_prev_dist.pop(unit_tag, None)
            self._prev_enemy_in_attack_range.pop(unit_tag, None)
            self._micro_under_fire_min_dist_prev.pop(unit_tag, None)
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            self._prev_kiting_enemy_tag.pop(unit_tag, None)
            return

        dist = float(unit.distance_to(enemy))
        enemy_range = self._weapon_range_attacker_vs_defender(enemy, unit)
        effective_enemy_range = max(
            enemy_range,
            self._get_unit_radius(enemy) + self._get_unit_radius(unit),
        )
        eps = 1e-3
        self._kiting_simple_prev_dist[unit_tag] = dist
        self._prev_enemy_in_attack_range[unit_tag] = dist <= effective_enemy_range + eps
        self._micro_under_fire_min_dist_prev[unit_tag] = self._min_dist_to_shooting_enemies_in_fovea(unit)
        self._prev_kiting_enemy_tag[unit_tag] = e_tag

    def calculate_kiting_simple_shaping(self, unit_tag: int) -> float:
        """
        Shaping for kiting:
        - Retreat reward (proportional): moving away when the assigned enemy was within its
          own attack range of this unit last step, AND the unit is still within its own weapon
          range. The weapon-range cap prevents the agent earning retreat reward indefinitely
          while fleeing to the map edge — retreat is only meaningful while the unit can still
          shoot back. Does NOT require the previous action to be an attack, so it fires in
          multi-unit cycling where several other units' steps separate attack and retreat.
        - Cooldown-retreat reward: moving away while this unit's weapon is on cooldown,
          while staying close enough to keep fighting. This rewards the actual kite rhythm
          (shoot, back up during cooldown, shoot again) before a melee enemy reaches contact.
        - Attack-on-cooldown penalty: discourages repeatedly issuing attack commands while
          the weapon cannot fire yet.
        - Close-in penalty (proportional): penalises moving closer to a currently-shooting
          enemy by `kiting_close_in_penalty * delta_distance_closer`.
        - Sweet-spot bonus: per-step reward for each enemy the unit can shoot (within own
          weapon range) while staying outside that enemy's attack range. Directly incentivises
          the kiting geometry: attack from distance, stay out of danger.
        - In-danger penalty: per-step cost for each enemy whose danger zone currently contains
          this unit. Discourages standing still inside enemy attack range.
        - Weapon-ready bonus (optional, set to 0 to disable): fires when the attack action
          is issued with weapon cooldown ≈ 0.
        """
        if unit_tag is None:
            return 0.0

        unit = self.units.find_by_tag(unit_tag)
        if unit is None:
            self._kiting_simple_prev_dist.pop(unit_tag, None)
            self._prev_enemy_in_attack_range.pop(unit_tag, None)
            return 0.0

        rewards_conf = self.training_conf.get("rewards", {})
        retreat_scale = float(rewards_conf.get("retreat_shaping_scale", 0.15))
        close_penalty = float(rewards_conf.get("kiting_close_in_penalty", 0.0))
        weapon_ready_bonus = float(rewards_conf.get("weapon_ready_attack_bonus", 0.0))
        sweet_spot_bonus = float(rewards_conf.get("sweet_spot_bonus", 0.0))
        in_danger_penalty = float(rewards_conf.get("in_danger_penalty", 0.0))
        cooldown_retreat_scale = float(rewards_conf.get("cooldown_retreat_scale", 0.0))
        attack_on_cooldown_penalty = float(rewards_conf.get("attack_on_cooldown_penalty", 0.0))

        self.refresh_combat_assignments()
        e_tag = self.current_assignments.get(unit_tag)
        enemies = self.enemy_units(self.enemy_unit_type)
        enemy = enemies.find_by_tag(e_tag) if e_tag is not None and enemies.exists else None
        if enemy is None:
            self._kiting_simple_prev_dist.pop(unit_tag, None)
            self._prev_enemy_in_attack_range.pop(unit_tag, None)
            self._prev_kiting_enemy_tag.pop(unit_tag, None)
            return 0.0

        # Guard: if the assignment changed since record_pre_action_kiting_state was called
        # (e.g. an enemy died during the step), the pre-action snapshot is stale — clear it
        # so we don't compare distances to two different enemies.
        prev_kiting_tag = self._prev_kiting_enemy_tag.get(unit_tag)
        if prev_kiting_tag is not None and prev_kiting_tag != e_tag:
            self._kiting_simple_prev_dist.pop(unit_tag, None)
            self._prev_enemy_in_attack_range.pop(unit_tag, None)
            self._micro_under_fire_min_dist_prev.pop(unit_tag, None)
            # _prev_kiting_enemy_tag is updated by record_pre_action_kiting_state, not here.

        dist = float(unit.distance_to(enemy))
        ally_range = self._weapon_range_attacker_vs_defender(unit, enemy)
        eps = 1e-3

        # Pre-action baseline recorded by record_pre_action_kiting_state (single-step).
        prev_dist = self._kiting_simple_prev_dist.get(unit_tag)
        prev_enemy_in_range = self._prev_enemy_in_attack_range.get(unit_tag, False)
        pre_action_cd = getattr(self, "_pre_action_weapon_cd_by_unit", {}).get(unit_tag)

        moved_now = bool(self.last_action_was_move_by_unit.get(unit_tag, False))
        attacked_now = bool(self.last_action_was_attack_by_unit.get(unit_tag, False))

        # Compute current min-shooting-enemy distance for close-penalty.
        curr_under = self._min_dist_to_shooting_enemies_in_fovea(unit)

        reward = 0.0

        # Retreat reward: moving away when the enemy was in contact range at step START
        # and the unit is still within its own weapon range. Using the pre-action baseline
        # (prev_dist = dist before apply_action) makes this a single-step comparison,
        # correctly measuring what the agent's move action achieved regardless of how many
        # other units acted between this unit's consecutive turns.
        if retreat_scale != 0.0 and moved_now and prev_enemy_in_range and prev_dist is not None:
            if dist > prev_dist + eps and dist <= ally_range + eps:
                reward += retreat_scale * (dist - prev_dist)

        # Cooldown-retreat reward: after firing, the correct kiting move is often to
        # back up before the enemy reaches contact range. This uses the same
        # single-step distance baseline as retreat shaping, but gates on weapon
        # cooldown instead of enemy contact.
        if (
            cooldown_retreat_scale != 0.0
            and ally_range > 1.5
            and moved_now
            and pre_action_cd is not None
            and float(pre_action_cd) > eps
            and prev_dist is not None
            and prev_dist <= ally_range + eps
            and dist > prev_dist + eps
            and dist <= ally_range + eps
        ):
            reward += cooldown_retreat_scale * (dist - prev_dist)

        # Close-in penalty (proportional): penalise moving closer to a currently-shooting enemy.
        if close_penalty != 0.0:
            prev_under = self._micro_under_fire_min_dist_prev.get(unit_tag)
            if (
                moved_now
                and prev_under is not None
                and curr_under is not None
                and float(curr_under) < float(prev_under) - eps
            ):
                delta_closer = float(prev_under) - float(curr_under)
                reward -= close_penalty * delta_closer

        # Weapon-ready attack bonus: fire exactly when cooldown has expired.
        if weapon_ready_bonus != 0.0 and attacked_now:
            cd_before = self._attack_order_weapon_cd_before.get(unit_tag)
            if cd_before is not None and float(cd_before) <= eps:
                reward += weapon_ready_bonus

        # Attack commands during cooldown usually just make the unit stand and wait,
        # which is the anti-kiting behavior we want the policy to avoid.
        if attack_on_cooldown_penalty != 0.0 and attacked_now:
            cd_before = self._attack_order_weapon_cd_before.get(unit_tag, pre_action_cd)
            if cd_before is not None and float(cd_before) > eps:
                reward -= attack_on_cooldown_penalty

        # Sweet-spot bonus: reward attacking from the kiting geometry (in range to shoot,
        # out of enemy range). Gated on attacked_now so it rewards the action of firing
        # from sweet spot rather than passive positioning at that distance.
        if sweet_spot_bonus != 0.0 and attacked_now:
            sweet_spots = self._count_kiting_sweet_spots_for_unit(unit)
            reward += sweet_spot_bonus * sweet_spots

        # In-danger penalty: cost for each enemy that can currently shoot this unit.
        if in_danger_penalty != 0.0:
            danger_zones = self._count_enemy_danger_zones_for_unit(unit)
            reward -= in_danger_penalty * danger_zones

        # State dicts (_kiting_simple_prev_dist, _prev_enemy_in_attack_range,
        # _micro_under_fire_min_dist_prev, _prev_kiting_enemy_tag) are populated by
        # record_pre_action_kiting_state() before the NEXT step — not updated here.
        self._attack_order_weapon_cd_before.pop(unit_tag, None)
        if hasattr(self, "_pre_action_weapon_cd_by_unit"):
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)

        return reward

    def calculate_use_global_reward(self, unit_tag: int) -> float:
        """
        Bonus for choosing move_global when no enemy is within
        ``_USE_GLOBAL_NO_ENEMY_RADIUS`` of the acting unit (pre-action snapshot).
        """
        if unit_tag is None:
            return 0.0

        rewards_conf = self.training_conf.get("rewards", {})
        use_global_bonus = float(rewards_conf.get("use_global", 0.0))
        if use_global_bonus == 0.0:
            return 0.0

        if self.last_move_head_by_unit.get(unit_tag) != "move_global":
            return 0.0

        if self._pre_action_enemy_within_use_global_radius.get(unit_tag, True):
            return 0.0

        return use_global_bonus

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

        # Always use the live assigned-enemy position as the command target so the
        # COMMAND_OBJECTIVE observation layer tracks actual enemy locations.  The static
        # spawn-anchor (_group_anchors) was stale once enemies moved, causing the
        # observation to point to an empty area of the map during active combat.
        target = self._get_enemy_target_for_unit(unit)
        target_pos = target.position if target else None

        raw_hc = self.training_conf.get("high_level_command", 0)
        if isinstance(raw_hc, str):
            high_cmd = CommandConstants.get_index(raw_hc)
        else:
            high_cmd = int(raw_hc)

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

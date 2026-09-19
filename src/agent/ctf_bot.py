from typing import Dict, Optional, Any, Tuple, List, Iterable, Set
import math
import numpy as np
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.data import Race
from loguru import logger

from src.hierarchical_agent import HierarchicalAgentSystem
from src.utils.layer_constants import LayerConstants
from src.utils.unit_feature_helpers import (
    _hostile_candidates,
    allied_candidates_for_owner,
    alliance_perspective_groups,
    counts_for_combat_hp_reward,
    first_unit,
    first_unit_of_type,
    heal_click_cell_half,
    heal_click_hits_unit,
    HEAL_CAST_RANGE,
    is_force_melee_unit,
    issue_aoe_at_point,
    issue_attack_at_point,
    perspective_unit_groups,
)
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

# Per-step TensorBoard command mix (trainee tags only; see env _record_command_metrics).
COMMAND_METRIC_KINDS: Tuple[str, ...] = ("move", "attack", "cloak", "uncloak")


def _missing_health(unit: Any) -> float:
    """HP missing from *unit*; mocks without numeric bars count as wounded."""
    try:
        health = float(getattr(unit, "health", 0.0) or 0.0)
        health_max = float(getattr(unit, "health_max", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 1.0
    if health_max <= 0.0:
        return 1.0
    return max(0.0, health_max - health)


class CaptureTheFlagBot(BotAI):
    """
    Bot for Capture the Flag environment.
    Supports individual unit control and reporting state to the environment.
    """
    # Matches Level7Bot / ObservationTransformer tactical crop (32×32 centered on unit).
    _TACTICAL_FOVEA_MAX_DELTA: float = 17.0
    # Weapon range at or below this is treated as melee (no cooldown-kite shaping).
    _RANGED_WEAPON_RANGE_MIN: float = 1.5
    # python-sc2 BotAI.time: Faster speed is 22.4 game loops per second.
    _HEAL_GAME_LOOPS_PER_SECOND: float = 22.4
    # Queen Transfuse cast is 1s (cooldown 1s / 0.71s Faster UI). Heal/Repair
    # lock-on is the same budget; do not hold the whole channel until idle.
    _HEAL_CAST_ANIMATION_SECONDS: Dict[UnitTypeId, float] = {
        UnitTypeId.QUEEN: 1.0,
        UnitTypeId.MEDIVAC: 1.0,
        UnitTypeId.SCV: 1.0,
    }
    _DEFAULT_HEAL_CAST_ANIMATION_SECONDS: float = 1.0
    _DEFAULT_HEALER_MOVE_SPEED: Dict[UnitTypeId, float] = {
        UnitTypeId.QUEEN: 1.31,
        UnitTypeId.MEDIVAC: 3.5,
        UnitTypeId.SCV: 3.94,
    }

    def __init__(self, player_id: int, training_conf: Optional[Dict] = None):
        super().__init__()
        self.player_id = player_id
        self.training_conf = training_conf or {}
        
        env_conf = self.training_conf.get("env", {})
        self.grid_size = self.training_conf.get("grid_size", env_conf.get("grid_size", 128))
        
        self.agent_system: Optional[HierarchicalAgentSystem] = None
        
        self.active_unit_tag: Optional[int] = None
        self.enemy_pylon_pos: Dict[int, Optional[Point2]] = {1: None, 2: None}
        
        # Asymmetric Self-Play attributes
        self.is_trainee = True # Default to trainee
        self.model = None # Snapshot model for automated inference
        
        # State tracking for rewards
        self.prev_pylon_health = {1: 20.0, 2: 20.0} # Combined HP + Shield
        self.prev_enemy_units_health = {1: 0.0, 2: 0.0}
        self.prev_own_units_health = {1: 0.0, 2: 0.0}
        self.prev_enemy_unit_count: Dict[int, int] = {1: 0, 2: 0}
        self.prev_own_unit_count: Dict[int, int] = {1: 0, 2: 0}
        # tag → (hp+shield, last position) for defender-zone combat credit
        self.prev_enemy_combat_snapshot: Dict[int, Dict[int, Tuple[float, Optional[Point2]]]] = {
            1: {},
            2: {},
        }
        self.prev_dist_to_pylon = {1: None, 2: None}
        self.prev_dist_per_unit: Dict[int, Optional[float]] = {}
        # Squad CoM distance tracking: (player_id, role) → float distance
        # Updated whenever that role's units are the active squad.
        self.prev_dist_per_squad: Dict[Tuple[int, str], Optional[float]] = {}

        # Maps unit tag → "attack" or "defend" — populated by the env after
        # region-based spawning so calculate_reward can apply role-specific
        # per-step signals.  Falls back to "attack" for unknown tags.
        self.unit_command_map: Dict[int, str] = {}

        # Maps unit tag → explicit objective Point2 — set at spawn time from
        # region pair data.  For attackers: the enemy defender's pylon position
        # in their arena.  For defenders: their own pylon position in their arena.
        # Both map to the _d center of the assigned pair.
        self.unit_target_map: Dict[int, Any] = {}

        # Maps unit tag → (head_name, Point2 target) of the last decoded SC2 order.
        # Re-issued every game step to non-active units so they never sit idle
        # while waiting for their turn in the cycling round-robin.
        self._unit_last_order: Dict[int, Tuple[str, Any]] = {}

        # Cooldown-retreat shaping — populated by record_pre_action_retreat_state()
        # before apply_action so the delta is single-step accurate in squad cycling.
        self._pre_action_weapon_cd_by_unit: Dict[int, float] = {}
        self._pre_action_dist_to_enemy_com: Dict[int, float] = {}
        self._pre_action_dist_to_defend_pylon: Dict[int, float] = {}
        self.last_action_was_move_by_unit: Dict[int, bool] = {}
        # Healer tags where this step's heal/repair/transfuse is a genuinely
        # *new* engagement (new patient, or the previous lock lapsed).
        # heal_cast_reward only pays when a selected tag is in this set, so a
        # camped re-issue on the same patient does not re-earn the cast bonus
        # every step. Cleared after reward.
        self._heal_cast_tags: Set[int] = set()
        # Healer tags that successfully issued repair/heal/transfuse this step,
        # fresh cast OR a continuing walk-in/channel on the same patient.
        # heal_reward (the per-HP shaping) pays off this broader set — a
        # Medivac's Heal beam or an SCV's Repair restores HP over many steps
        # after the *one* step that starts the cast; gating HP credit on
        # "fresh cast only" (like heal_cast_reward) would zero out nearly all
        # of a channeled heal's reward, since HP has usually not ticked yet on
        # the single step that first issues the order. Not battery ticks, Heal
        # autocast, or Medivac attack-move HP — those never populate this set.
        # Cleared after reward.
        self._heal_order_tags: Set[int] = set()
        # Selected tags that received a heal_* command this step (hit or miss).
        self._heal_attempt_tags: Set[int] = set()
        # Selected tags that received move / attack / cloak / uncloak this step.
        self._command_attempt_tags: Dict[str, Set[int]] = {
            kind: set() for kind in COMMAND_METRIC_KINDS
        }
        # Sum of cooldown-retreat shaping paid this physics step (TensorBoard).
        self._step_cooldown_retreat_reward: float = 0.0
        # healer_tag → (patient_tag, expiry_game_loop) while walk-in + cast
        # animation must not be overwritten by attack/move.
        self._heal_lock: Dict[int, Tuple[int, int]] = {}
        # healer_tag → game_loop when Transfuse/Heal cooldown ends (P2 fallback).
        self._heal_ability_ready_after_loop: Dict[int, int] = {}
        # tag → set(AbilityId) from the latest SC2 ability query.
        self._available_abilities_by_tag: Dict[int, Any] = {}
        
        self.frame_count = 0
        self.cached_observation: Optional[Any] = None

        # enemy_race must be set to avoid AttributeError in BotAIInternal._prepare_step
        self._enemy_race = Race.Random
        
        # Satisfaction for BotAI initialization
        self._initialize_variables()

    def count_heal_casts_for_tags(self, squad_tags: Iterable) -> int:
        """How many selected units issued Repair / Heal / Transfuse this step."""
        return self._count_heal_tags(getattr(self, "_heal_cast_tags", None), squad_tags)

    def count_heal_attempts_for_tags(self, squad_tags: Iterable) -> int:
        """How many selected units were given a heal_* command this step."""
        return self._count_heal_tags(getattr(self, "_heal_attempt_tags", None), squad_tags)

    def count_command_attempts_for_tags(self, kind: str, squad_tags: Iterable) -> int:
        """How many selected units were given a move/attack/cloak/uncloak this step."""
        buckets = getattr(self, "_command_attempt_tags", None)
        if not isinstance(buckets, dict):
            return 0
        return self._count_heal_tags(buckets.get(kind), squad_tags)

    @staticmethod
    def _count_heal_tags(tag_set: Any, squad_tags: Iterable) -> int:
        if not isinstance(tag_set, set) or not squad_tags:
            return 0
        return sum(1 for tag in squad_tags if int(tag) in tag_set)

    def _mark_heal_command_attempt(self, unit: Unit) -> None:
        tag = int(getattr(unit, "tag", 0) or 0)
        if tag <= 0:
            return
        attempts = getattr(self, "_heal_attempt_tags", None)
        if not isinstance(attempts, set):
            self._heal_attempt_tags = set()
            attempts = self._heal_attempt_tags
        attempts.add(tag)

    def _mark_command_attempt(self, unit: Unit, kind: str) -> None:
        """Record that ``unit`` received a move/attack/cloak/uncloak order this step."""
        if kind not in COMMAND_METRIC_KINDS:
            return
        tag = int(getattr(unit, "tag", 0) or 0)
        if tag <= 0:
            return
        buckets = getattr(self, "_command_attempt_tags", None)
        if not isinstance(buckets, dict):
            self._command_attempt_tags = {k: set() for k in COMMAND_METRIC_KINDS}
            buckets = self._command_attempt_tags
        bucket = buckets.get(kind)
        if not isinstance(bucket, set):
            bucket = set()
            buckets[kind] = bucket
        bucket.add(tag)

    def _clear_command_attempt_tags(self) -> None:
        buckets = getattr(self, "_command_attempt_tags", None)
        if not isinstance(buckets, dict):
            self._command_attempt_tags = {k: set() for k in COMMAND_METRIC_KINDS}
            return
        for kind in COMMAND_METRIC_KINDS:
            bucket = buckets.get(kind)
            if isinstance(bucket, set):
                bucket.clear()
            else:
                buckets[kind] = set()

    def reset_episode_state(self):
        """Reset all per-episode tracking so reward baselines don't bleed across episodes."""
        self.active_unit_tag = None
        self.enemy_pylon_pos = {1: None, 2: None}
        self.prev_pylon_health = {1: 20.0, 2: 20.0}
        self.prev_enemy_units_health = {1: 0.0, 2: 0.0}
        self.prev_own_units_health = {1: 0.0, 2: 0.0}
        self.prev_enemy_unit_count: Dict[int, int] = {1: 0, 2: 0}
        self.prev_own_unit_count: Dict[int, int] = {1: 0, 2: 0}
        self.prev_enemy_combat_snapshot = {1: {}, 2: {}}
        self.prev_dist_to_pylon = {1: None, 2: None}
        self.prev_dist_per_unit = {}
        self.unit_command_map = {}
        self.unit_target_map = {}
        self._unit_last_order = {}
        self.prev_dist_per_squad = {}
        self._pre_action_weapon_cd_by_unit = {}
        self._pre_action_dist_to_enemy_com = {}
        self._pre_action_dist_to_defend_pylon = {}
        self.last_action_was_move_by_unit = {}
        self._heal_cast_tags = set()
        self._heal_order_tags = set()
        self._heal_attempt_tags = set()
        self._command_attempt_tags = {kind: set() for kind in COMMAND_METRIC_KINDS}
        self._step_cooldown_retreat_reward = 0.0
        self._heal_lock = {}
        self._heal_ability_ready_after_loop = {}
        self._available_abilities_by_tag = {}
        self.cached_observation = None
        self.frame_count = 0

    def pop_cooldown_retreat_reward(self) -> float:
        """Return and clear the cooldown-retreat shaping paid this physics step."""
        total = float(getattr(self, "_step_cooldown_retreat_reward", 0.0) or 0.0)
        self._step_cooldown_retreat_reward = 0.0
        return total

    def prime_combat_reward_baselines(self) -> None:
        """Snapshot army/pylon HP after spawn so the first physics step is not a fake heal."""
        for p_id in (1, 2):
            own_units, enemy_units, own_structs, enemy_structs = self._perspective_unit_groups(
                p_id
            )
            self.prev_own_units_health[p_id] = self._mobile_combat_hp_sum(
                own_units, include_filter=counts_for_combat_hp_reward
            )
            self.prev_enemy_units_health[p_id] = self._mobile_combat_hp_sum(
                enemy_units, include_filter=counts_for_combat_hp_reward
            )
            self.prev_own_unit_count[p_id] = self._mobile_combat_count(
                own_units, include_filter=counts_for_combat_hp_reward
            )
            self.prev_enemy_unit_count[p_id] = self._mobile_combat_count(
                enemy_units, include_filter=counts_for_combat_hp_reward
            )
            self.prev_enemy_combat_snapshot[p_id] = self._combat_snapshot(
                enemy_units, include_filter=counts_for_combat_hp_reward
            )

            own_pylon = self._first_pylon(own_structs)
            enemy_pylon = self._first_pylon(enemy_structs)
            if own_pylon is None or enemy_pylon is None:
                _, _, bucket_own, bucket_en = self._alliance_perspective_groups(p_id)
                if own_pylon is None:
                    own_pylon = self._first_pylon(bucket_own)
                if enemy_pylon is None:
                    enemy_pylon = self._first_pylon(bucket_en)
            own_hp = 0.0
            enemy_hp = 0.0
            if own_pylon is not None:
                own_hp = float(own_pylon.health) + float(getattr(own_pylon, "shield", 0.0) or 0.0)
            if enemy_pylon is not None:
                enemy_hp = float(enemy_pylon.health) + float(
                    getattr(enemy_pylon, "shield", 0.0) or 0.0
                )
            if p_id == 1:
                self.prev_pylon_health[1] = own_hp
                if enemy_pylon is not None:
                    self.prev_pylon_health[2] = enemy_hp
            else:
                self.prev_pylon_health[2] = own_hp
                if enemy_pylon is not None:
                    self.prev_pylon_health[1] = enemy_hp

    def _get_zero_observation(self) -> Any:
        """Shape-correct empty observation (dict for universal, array for original)."""
        num_layers = len(self.training_conf.get("active_layers", [])) or LayerConstants.get_num_layers()
        if self.training_conf.get("agent_architecture") == "universal":
            n_tactical = len(self.training_conf.get("active_tactical_layers", [])) or num_layers
            return {
                "map": np.zeros((num_layers, 128, 128), dtype=np.float32),
                "tactical": np.zeros((n_tactical, 32, 32), dtype=np.float32),
                "command": np.zeros(10, dtype=np.float32),
            }
        return np.zeros((num_layers, 128, 128), dtype=np.float32)

    async def on_start(self):
        env_conf = self.training_conf.get("env", {})
        arch = self.training_conf.get("agent_architecture", "original")
        if arch in ("universal", "squad"):
            # Match SC2BotUniversal / MultiDiscrete 32×32 spatial head (not legacy 128).
            self.grid_size = 32
        else:
            self.grid_size = self.training_conf.get(
                "grid_size", env_conf.get("grid_size", 128)
            )

        decode_conf = dict(self.training_conf)
        if arch == "squad":
            decode_conf["agent_architecture"] = "universal"

        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=self.grid_size,
            training_conf=decode_conf,
        )
        self.agent_system.create_unit_agent("active_agent")

    def get_observation(self, perspective_player_id: Optional[int] = None) -> Any:
        """Returns the processed observation from the given player's perspective."""
        # Use provided perspective or default to bot's own player_id
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id
        
        # Defensive check for game info/state
        if not hasattr(self, "game_info") or self.game_info is None:
             logger.warning(f"Bot {self.player_id}: game_info is missing. Returning zero observation.")
             return self._get_zero_observation()

        # Owner-id first (god-mode mixes alliance buckets). Trainee may be P2.
        units, enemy_units, structures, enemy_structures = self._perspective_unit_groups(
            p_id
        )

        enemy_pylon = self._first_pylon(enemy_structures)

        # Persistent enemy-pylon position tracking (fog-of-war fallback)
        if enemy_pylon:
            self.enemy_pylon_pos[p_id] = enemy_pylon.position

        active_unit = None
        selected_units = []
        if self.active_unit_tag:
            try:
                active_unit = self.all_units.find_by_tag(self.active_unit_tag)
                if active_unit:
                    selected_units = [active_unit]
            except Exception:
                pass
        if active_unit is None:
            active_unit = first_unit(units)
            if active_unit is not None:
                selected_units = [active_unit]
                self.active_unit_tag = active_unit.tag

        # command_target: use per-unit spawn-time objective when available.
        # For attackers this is the enemy defender's pylon in their arena;
        # for defenders it is their own pylon in their arena (both = _d center).
        # Fall back to runtime pylon tracking when the map is unknown (e.g. legacy path).
        if self.active_unit_tag and self.active_unit_tag in self.unit_target_map:
            target_pos = self.unit_target_map[self.active_unit_tag]
        else:
            unit_role = self.unit_command_map.get(self.active_unit_tag) if self.active_unit_tag else None
            if unit_role == "defend":
                own_pylon = self._first_pylon(structures)
                target_pos = own_pylon.position if own_pylon else None
            else:
                target_pos = self.enemy_pylon_pos[p_id]

        from src.utils.command_constants import CommandConstants

        raw_hc = self.training_conf.get("high_level_command", 0)
        if self.active_unit_tag and self.active_unit_tag in self.unit_command_map:
            raw_hc = self.unit_command_map[self.active_unit_tag]
        high_cmd = CommandConstants.get_index(raw_hc)

        raw_obs = {
            'units': units,
            'enemy_units': enemy_units,
            'enemy_structures': enemy_structures,
            'structures': structures,
            'effects': getattr(self.state, "effects", []),
            'game_info': self.game_info,
            'game_data': self.game_data,
            'creep': getattr(self.state, "creep", None),
            'command_target': target_pos,
            'selected_units': selected_units,
            'high_level_command': high_cmd,
        }

        # Process with agent system, passing active_unit for the tactical crop
        if self.agent_system is None:
            logger.warning(f"Bot {self.player_id}: agent_system not ready. Returning zero observation.")
            return self._get_zero_observation()
        try:
            processed = self.agent_system.process_observation(
                raw_obs, player_id=p_id, active_unit=active_unit
            )
            obs = processed["active_agent"]
            # Perspective is ownership only (allied vs enemy). Maps stay in
            # world coordinates so a P2 trainee matches the pre-leak-fix P1 seat.
            self.cached_observation = obs
        except Exception as e:
            if self.cached_observation is not None:
                logger.warning(f"Bot {self.player_id} (P_ID {p_id}): Observation processing failed ({e}). Returning cached observation.")
                return self.cached_observation
            else:
                logger.warning(
                    f"Bot {self.player_id} (P_ID {p_id}): Observation processing failed ({e}). "
                    "Returning zero observation."
                )
                return self._get_zero_observation()

        return self.cached_observation

    def predict_action(self) -> int:
        """
        Uses the internal snapshot model to predict an action for the current state.
        Only used if self.is_trainee is False.
        """
        if self.is_trainee:
            raise RuntimeError("Trainee bot should not predict actions; it receives them from the external trainer.")
        
        if self.model is None:
            logger.warning(f"Bot {self.player_id}: No model set for automated inference. Returning default action 0.")
            return 0

        obs = self.get_observation()
        masks = self.action_masks()
        
        # SB3 models expect a batch dimension [1, ...]
        obs_batch = np.expand_dims(obs, axis=0)
        
        # Inference (deterministic=True for stable opponent behavior)
        action, _states = self.model.predict(obs_batch, action_masks=np.expand_dims(masks, axis=0), deterministic=True)
        
        # action is typically a numpy array [action_value]
        if isinstance(action, np.ndarray):
            if action.ndim > 0:
                return int(action[0])
            return int(action)
        return int(action)

    def action_masks(self) -> np.ndarray:
        """
        Universal MultiDiscrete masking: flat [command | spatial].

        Mirrors SC2BotUniversal.action_masks — only unconfigured (padded)
        command slots are masked False; all spatial cells are open.
        """
        action_heads = self.training_conf.get("action_heads") or []
        categorical_heads = self.training_conf.get("categorical_heads") or []
        n_configured = min(
            FIXED_UNIVERSAL_COMMAND_DIM,
            len(action_heads) + len(categorical_heads),
        )
        cmd_mask = np.zeros(FIXED_UNIVERSAL_COMMAND_DIM, dtype=bool)
        cmd_mask[:n_configured] = True
        spatial_mask = np.ones(32 * 32, dtype=bool)
        return np.concatenate([cmd_mask, spatial_mask])

    def squad_action_masks(self, unit_mask: np.ndarray) -> np.ndarray:
        """Flat mask for squad pointer + command + spatial action vector."""
        from src.agent.squad_constants import (
            MAX_SQUAD_UNITS,
            squad_action_space_nvec,
        )
        from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

        nvec = squad_action_space_nvec()
        flat_len = int(sum(nvec))
        mask = np.ones(flat_len, dtype=bool)
        unit_mask = np.asarray(unit_mask, dtype=np.float32).reshape(-1)
        for i in range(min(MAX_SQUAD_UNITS, len(unit_mask))):
            if unit_mask[i] < 0.5:
                # Dead/padded slot: force "not selected". Masking BOTH options False
                # makes MaskableCategorical fall back to uniform 50/50 sampling.
                mask[i * 2] = True
                mask[i * 2 + 1] = False
        action_heads = list(self.training_conf.get("action_heads") or [])
        categorical_heads = list(self.training_conf.get("categorical_heads") or [])
        configured = action_heads + categorical_heads
        n_configured = min(FIXED_UNIVERSAL_COMMAND_DIM, len(configured))
        cmd_start = MAX_SQUAD_UNITS * 2
        mask[cmd_start : cmd_start + FIXED_UNIVERSAL_COMMAND_DIM] = False
        mask[cmd_start : cmd_start + n_configured] = True
        return mask

    async def apply_squad_action(
        self,
        selected_tags: List[int],
        *,
        anchor_tag: int,
        command_idx: int,
        spatial_idx: int,
        perspective_player_id: int,
    ) -> None:
        """Apply one squad group order to selected units (shared world target).

        ``perspective_player_id`` is the env's SC2 owner for this order (trainee
        or opponent). Spatial cells are world-frame for every owner: no 180°
        un-rotation. Targeting uses each caster's ``owner_id``.
        """
        if not self.agent_system or "active_agent" not in self.agent_system.unit_agents:
            return

        action_heads = list(self.training_conf.get("action_heads") or [])
        categorical_heads = list(self.training_conf.get("categorical_heads") or [])
        n_spatial = len(action_heads)
        n_total = n_spatial + len(categorical_heads)
        if command_idx < 0 or command_idx >= n_total:
            return

        is_categorical = command_idx >= n_spatial
        if is_categorical:
            head_name = categorical_heads[command_idx - n_spatial]
        else:
            head_name = action_heads[command_idx]

        anchor = self.all_units.find_by_tag(anchor_tag)
        if anchor is None and selected_tags:
            anchor = self.all_units.find_by_tag(selected_tags[0])
        if anchor is None:
            return

        # CoM / role shaping in calculate_squad_step_reward keys off active_unit_tag.
        self.active_unit_tag = int(anchor.tag)

        agent = self.agent_system.unit_agents["active_agent"]
        game_info = getattr(self, "game_info", None)
        if game_info is not None:
            agent.last_game_info = game_info

        # Categorical cloak/burrow: no spatial decode.
        if is_categorical and head_name in ("cloak_burrow", "uncloak_unburrow"):
            conceal = head_name == "cloak_burrow"
            metric_kind = "cloak" if conceal else "uncloak"
            for unit_tag in selected_tags:
                unit = self.all_units.find_by_tag(unit_tag)
                if unit is None or unit.type_id == UnitTypeId.SHIELDBATTERY:
                    continue
                self._mark_command_attempt(unit, metric_kind)
                self._issue_cloak_burrow(unit, conceal=conceal)
                self._unit_last_order[int(unit_tag)] = (head_name, None)
                self.last_action_was_move_by_unit[int(unit_tag)] = False
            return

        if is_categorical:
            return  # Unknown categorical head — no-op

        decode_unit = anchor
        world_x, world_y = agent.convert_spatial_action_to_world(
            head_name, int(spatial_idx), active_unit=decode_unit
        )
        target = Point2((float(world_x), float(world_y)))
        if selected_tags and hasattr(agent, "record_debug_click"):
            agent.record_debug_click(target, decode_unit)

        for unit_tag in selected_tags:
            unit = self.all_units.find_by_tag(unit_tag)
            if unit is None:
                continue
            if unit.type_id == UnitTypeId.SHIELDBATTERY:
                # Auto-heals; never issue move/attack/heal orders.
                continue
            if "heal" in head_name:
                self._mark_heal_command_attempt(unit)
                self._issue_heal_at_point(unit, target)
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                # Misses are no-ops, not moves: cooldown-retreat must not treat
                # a failed Transfuse as a kite.
                self.last_action_was_move_by_unit[int(unit_tag)] = False
            elif self._refresh_heal_lock(unit):
                # Walk-in + Transfuse/Heal start still inside the lock window.
                continue
            elif "aoe" in head_name:
                issue_aoe_at_point(self, unit, target)
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self.last_action_was_move_by_unit[int(unit_tag)] = False
            elif "unload" in head_name:
                self._issue_unload_at_point(unit, target)
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self.last_action_was_move_by_unit[int(unit_tag)] = False
            elif "load" in head_name:
                self._issue_load_at_point(unit, target)
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self.last_action_was_move_by_unit[int(unit_tag)] = False
            elif "attack" in head_name:
                issue_attack_at_point(self, unit, target)
                self._mark_command_attempt(unit, "attack")
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self.last_action_was_move_by_unit[int(unit_tag)] = False
            else:
                self.do(unit.move(target))
                self._mark_command_attempt(unit, "move")
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self.last_action_was_move_by_unit[int(unit_tag)] = True

    def _allied_units_for_owner(self, owner_id: int) -> List[Unit]:
        """Allied units visible from God-Mode for a given owner."""
        return allied_candidates_for_owner(self, int(owner_id))

    @staticmethod
    def _unit_in_spatial_cell(unit: Unit, target: Point2, cell_half: float = 0.5) -> bool:
        """True when unit center lies in the world cell around ``target``."""
        return (
            abs(float(unit.position.x) - float(target.x)) < cell_half
            and abs(float(unit.position.y) - float(target.y)) < cell_half
        )

    @staticmethod
    def _heal_target_is_eligible(healer: Unit, ally: Unit) -> bool:
        """True when ``ally``'s biology matches what ``healer``'s ability can affect.

        Real SC2 silently ignores an order targeting the wrong category — the
        Medivac's Heal only affects Biological units, the SCV's Repair only
        affects Mechanical units (or structures, excluded above already).
        Without this check, a Medivac clicked onto a wounded Siege Tank (or an
        SCV onto a wounded Marine) resolves a target, credits
        heal_cast_reward, and sets the walk-in lock — but the in-game order
        never actually heals anything, so ``heal_reward``'s HP credit never
        follows and the click is not a real "finish" despite counting as a
        cast in our own metrics.
        """
        if healer.type_id in (UnitTypeId.MEDIVAC, UnitTypeId.QUEEN):
            return bool(getattr(ally, "is_biological", False))
        if healer.type_id == UnitTypeId.SCV:
            return bool(getattr(ally, "is_mechanical", False))
        return True

    def _resolve_heal_target_in_cell(self, healer: Unit, target: Point2) -> Optional[Unit]:
        """Wounded allied non-structure under the click.

        Same identity rule as attack: the click picks the unit. The engine walks
        into Transfuse / Heal / Repair range. Do not require the ally to already
        be in cast range (the tactical crop is 17, Queen Transfuse is 7 — that
        gate turned correct clicks into moves). Do not search for an ally near
        the click. Prefers the ally missing the most HP. A click hits the
        configured square (``env.heal_click_cells``; omit for 1×1) or the unit body.
        Skips allies the healer's specific ability cannot affect (see
        ``_heal_target_is_eligible``) — Heal is Biological-only, Repair is
        Mechanical-only.
        """
        best: Optional[Unit] = None
        best_missing = 0.0
        cell_half = heal_click_cell_half(self.training_conf)
        for ally in self._allied_units_for_owner(int(healer.owner_id)):
            if ally is None or getattr(ally, "tag", None) == healer.tag:
                continue
            if getattr(ally, "is_structure", False):
                continue
            if not self._heal_target_is_eligible(healer, ally):
                continue
            if not heal_click_hits_unit(ally, target, cell_half=cell_half):
                continue
            missing = _missing_health(ally)
            if missing <= 0.0:
                continue
            if missing > best_missing:
                best_missing = missing
                best = ally
        return best

    def _game_loop(self) -> int:
        """Current SC2 game loop, or 0 when the mock has no numeric clock."""
        state = getattr(self, "state", None)
        raw = getattr(state, "game_loop", None) if state is not None else None
        if raw is None:
            raw = getattr(self, "frame_count", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _healer_move_speed(unit: Unit) -> float:
        for attr in ("real_speed", "movement_speed"):
            try:
                speed = float(getattr(unit, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                speed = 0.0
            if speed > 1e-6:
                return speed
        defaults = CaptureTheFlagBot._DEFAULT_HEALER_MOVE_SPEED
        return float(defaults.get(getattr(unit, "type_id", None), 1.31))

    def _heal_lock_until_loop(self, healer: Unit, patient: Unit) -> int:
        """Game loop when walk-in + Transfuse/Heal start animation should finish.

        Queen Transfuse is a 1s cast, not a channel (the 7s HoT is on the
        patient). Medivac Heal / SCV Repair *are* channels; lock only the
        start window so combat is not frozen for the whole beam.
        """
        now = CaptureTheFlagBot._game_loop(self)
        type_id = getattr(healer, "type_id", None)
        anim = float(
            CaptureTheFlagBot._HEAL_CAST_ANIMATION_SECONDS.get(
                type_id, CaptureTheFlagBot._DEFAULT_HEAL_CAST_ANIMATION_SECONDS
            )
        )
        try:
            cast_range = float(HEAL_CAST_RANGE.get(type_id, 0.0) or 0.0)
        except (TypeError, ValueError):
            cast_range = 0.0
        extra = 0.0
        try:
            dist = float(healer.position.distance_to(patient.position))
            extra = max(0.0, dist - cast_range)
        except (TypeError, ValueError, AttributeError):
            extra = 0.0
        speed = CaptureTheFlagBot._healer_move_speed(healer)
        walk = extra / max(speed, 1e-6)
        loops = int(
            math.ceil((walk + anim) * CaptureTheFlagBot._HEAL_GAME_LOOPS_PER_SECOND)
        )
        return now + max(loops, 1)

    def _refresh_heal_lock(self, unit: Unit) -> bool:
        """True while walk-in + cast animation must not be overwritten.

        Queens often spend several physics steps walking into range 7. Issuing
        attack or move on the next squad step cancels that. Lock until the
        computed window elapses, the caster goes idle, or the patient dies.
        ``heal_local`` is still allowed (retarget).
        """
        tag = int(getattr(unit, "tag", 0) or 0)
        lock = getattr(self, "_heal_lock", None)
        if tag <= 0 or not isinstance(lock, dict) or tag not in lock:
            return False
        entry = lock[tag]
        try:
            patient_tag = int(entry[0])
            until_loop = int(entry[1])
        except (TypeError, ValueError, IndexError):
            lock.pop(tag, None)
            return False
        now = CaptureTheFlagBot._game_loop(self)
        if now >= until_loop:
            lock.pop(tag, None)
            return False
        if bool(getattr(unit, "is_idle", False)):
            lock.pop(tag, None)
            return False
        patient = self.all_units.find_by_tag(patient_tag)
        if patient is None:
            lock.pop(tag, None)
            return False
        return True

    def _issue_heal_at_point(self, unit: Unit, target: Point2) -> Optional[str]:
        """Heal the cell-local wounded ally, or no-op on a miss.

        Empty ground and full-HP allies issue no order. Attack still
        attack-moves on an empty cell; a missed heal is not rewritten as a
        walk, so cooldown-retreat and attacker center-of-mass distance do not
        credit ``heal_local`` as kiting or approaching. Out-of-range wounded
        allies still get the ability so the engine walks in. The click picks
        the patient; do not retarget to another selected unit.

        Returns ``\"cast\"`` when a heal/repair/transfuse was issued, or
        ``None`` for a miss or non-healer.
        """
        ability = None
        if unit.type_id == UnitTypeId.MEDIVAC:
            ability = AbilityId.MEDIVACHEAL_HEAL
        elif unit.type_id == UnitTypeId.SCV:
            ability = AbilityId.EFFECT_REPAIR_SCV
        elif unit.type_id == UnitTypeId.QUEEN:
            ability = AbilityId.TRANSFUSION_TRANSFUSION
        else:
            return None

        heal_target = self._resolve_heal_target_in_cell(unit, target)
        if heal_target is None:
            return None
        self.do(unit(ability, heal_target))
        tag = int(getattr(unit, "tag", 0) or 0)
        if tag:
            order_tags = getattr(self, "_heal_order_tags", None)
            if not isinstance(order_tags, set):
                self._heal_order_tags = set()
                order_tags = self._heal_order_tags
            order_tags.add(tag)

            lock = getattr(self, "_heal_lock", None)
            if not isinstance(lock, dict):
                self._heal_lock = {}
                lock = self._heal_lock
            patient_tag = int(getattr(heal_target, "tag", 0) or 0)

            # A reissue on the *same* patient while the previous cast's
            # walk-in/animation lock is still active is the engine reaffirming
            # one in-flight heal, not a new cast. Crediting heal_cast_reward
            # on every squad step the policy keeps clicking pays *per step
            # parked on heal*, not per cast landed — that let heal_local
            # out-earn combat just by camping a wounded ally and crowded out
            # attack_local. Only a fresh engagement (new patient, or the
            # previous lock lapsed) is cast_bonus-eligible; heal_reward's
            # per-HP shaping still uses the broader ``_heal_order_tags`` above
            # so a Medivac beam / SCV repair keeps earning HP credit on every
            # step it actually ticks HP, not just the step that started it.
            prev_entry = lock.get(tag)
            now = CaptureTheFlagBot._game_loop(self)
            continuing_same_patient = (
                prev_entry is not None
                and int(prev_entry[0]) == patient_tag
                and now < int(prev_entry[1])
            )
            if not continuing_same_patient:
                from src.utils.unit_feature_helpers import mark_heal_ability_cooldown

                mark_heal_ability_cooldown(self, unit)
                cast_tags = getattr(self, "_heal_cast_tags", None)
                if not isinstance(cast_tags, set):
                    self._heal_cast_tags = set()
                    cast_tags = self._heal_cast_tags
                cast_tags.add(tag)
            if patient_tag:
                lock[tag] = (
                    patient_tag,
                    CaptureTheFlagBot._heal_lock_until_loop(self, unit, heal_target),
                )
        return "cast"

    _TRANSPORT_LOAD_ABILITY = {
        UnitTypeId.MEDIVAC: AbilityId.LOAD_MEDIVAC,
        UnitTypeId.OVERLORD: AbilityId.LOAD_OVERLORD,
        UnitTypeId.OVERLORDTRANSPORT: AbilityId.LOAD_OVERLORD,
        UnitTypeId.WARPPRISM: AbilityId.LOAD_WARPPRISM,
        UnitTypeId.WARPPRISMPHASING: AbilityId.LOAD_WARPPRISM,
    }
    _TRANSPORT_UNLOAD_ABILITY = {
        UnitTypeId.MEDIVAC: AbilityId.UNLOADALLAT_MEDIVAC,
        UnitTypeId.OVERLORD: AbilityId.UNLOADALLAT_OVERLORD,
        UnitTypeId.OVERLORDTRANSPORT: AbilityId.UNLOADALLAT_OVERLORD,
        UnitTypeId.WARPPRISM: AbilityId.UNLOADALLAT_WARPPRISM,
        UnitTypeId.WARPPRISMPHASING: AbilityId.UNLOADALL_WARPPRISM,
    }

    def _resolve_loadable_in_cell(self, transport: Unit, target: Point2) -> Optional[Unit]:
        """Allied non-transport in the clicked cell (loadable passenger)."""
        for ally in self._allied_units_for_owner(int(transport.owner_id)):
            if ally is None or getattr(ally, "tag", None) == transport.tag:
                continue
            if getattr(ally, "is_structure", False):
                continue
            if int(getattr(ally, "cargo_max", 0) or 0) > 0:
                continue
            if self._unit_in_spatial_cell(ally, target):
                return ally
        return None

    def _resolve_transport_in_cell(self, passenger: Unit, target: Point2) -> Optional[Unit]:
        """Allied transport in the clicked cell."""
        for ally in self._allied_units_for_owner(int(passenger.owner_id)):
            if ally is None or getattr(ally, "tag", None) == passenger.tag:
                continue
            if int(getattr(ally, "cargo_max", 0) or 0) <= 0:
                continue
            if self._unit_in_spatial_cell(ally, target):
                return ally
        return None

    def _issue_load_at_point(self, unit: Unit, target: Point2) -> None:
        """Dual-role load: transport LOADs cell ally; non-transport smarts onto cell transport."""
        cargo_max = int(getattr(unit, "cargo_max", 0) or 0)
        if cargo_max > 0:
            ability = self._TRANSPORT_LOAD_ABILITY.get(unit.type_id)
            if ability is None:
                return
            loadable = self._resolve_loadable_in_cell(unit, target)
            if loadable is None:
                return
            self.do(unit(ability, loadable))
            return

        transport = self._resolve_transport_in_cell(unit, target)
        if transport is None:
            return
        self.do(unit.smart(transport))

    def _issue_unload_at_point(self, unit: Unit, target: Point2) -> None:
        """Unload all cargo at the decoded world point (transports with cargo only)."""
        if int(getattr(unit, "cargo_max", 0) or 0) <= 0:
            return
        if int(getattr(unit, "cargo_used", 0) or 0) <= 0:
            return
        ability = self._TRANSPORT_UNLOAD_ABILITY.get(unit.type_id, AbilityId.UNLOADALLAT)
        if ability == AbilityId.UNLOADALL_WARPPRISM:
            self.do(unit(ability))
        else:
            self.do(unit(ability, target))

    _BURROW_DOWN_BY_TYPE = {
        UnitTypeId.ZERGLING: AbilityId.BURROWDOWN_ZERGLING,
        UnitTypeId.BANELING: AbilityId.BURROWDOWN_BANELING,
        UnitTypeId.ROACH: AbilityId.BURROWDOWN_ROACH,
        UnitTypeId.RAVAGER: AbilityId.BURROWDOWN_RAVAGER,
        UnitTypeId.HYDRALISK: AbilityId.BURROWDOWN_HYDRALISK,
        UnitTypeId.INFESTOR: AbilityId.BURROWDOWN_INFESTOR,
        UnitTypeId.SWARMHOSTMP: AbilityId.BURROWDOWN_SWARMHOST,
        UnitTypeId.ULTRALISK: AbilityId.BURROWDOWN_ULTRALISK,
        UnitTypeId.QUEEN: AbilityId.BURROWDOWN_QUEEN,
        UnitTypeId.DRONE: AbilityId.BURROWDOWN_DRONE,
        UnitTypeId.WIDOWMINE: AbilityId.BURROWDOWN_WIDOWMINE,
        UnitTypeId.LURKERMP: AbilityId.BURROWDOWN_LURKER,
    }
    _BURROW_UP_BY_TYPE = {
        UnitTypeId.ZERGLINGBURROWED: AbilityId.BURROWUP_ZERGLING,
        UnitTypeId.BANELINGBURROWED: AbilityId.BURROWUP_BANELING,
        UnitTypeId.ROACHBURROWED: AbilityId.BURROWUP_ROACH,
        UnitTypeId.RAVAGERBURROWED: AbilityId.BURROWUP_RAVAGER,
        UnitTypeId.HYDRALISKBURROWED: AbilityId.BURROWUP_HYDRALISK,
        UnitTypeId.INFESTORBURROWED: AbilityId.BURROWUP_INFESTOR,
        UnitTypeId.SWARMHOSTBURROWEDMP: AbilityId.BURROWUP_SWARMHOST,
        UnitTypeId.ULTRALISKBURROWED: AbilityId.BURROWUP_ULTRALISK,
        UnitTypeId.QUEENBURROWED: AbilityId.BURROWUP_QUEEN,
        UnitTypeId.DRONEBURROWED: AbilityId.BURROWUP_DRONE,
        UnitTypeId.WIDOWMINEBURROWED: AbilityId.BURROWUP_WIDOWMINE,
        UnitTypeId.LURKERMPBURROWED: AbilityId.BURROWUP_LURKER,
        UnitTypeId.LURKERBURROWED: AbilityId.BURROWUP_LURKER,
    }

    def _issue_cloak_burrow(self, unit: Unit, *, conceal: bool) -> None:
        """Shared categorical head: cloak↔burrow / uncloak↔unburrow by unit type."""
        tid = unit.type_id
        if tid == UnitTypeId.DARKTEMPLAR:
            return  # permanently cloaked
        if tid == UnitTypeId.GHOST:
            ability = (
                AbilityId.BEHAVIOR_CLOAKON_GHOST
                if conceal
                else AbilityId.BEHAVIOR_CLOAKOFF_GHOST
            )
            self.do(unit(ability))
            return
        if tid == UnitTypeId.BANSHEE:
            ability = (
                AbilityId.BEHAVIOR_CLOAKON_BANSHEE
                if conceal
                else AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE
            )
            self.do(unit(ability))
            return
        if tid == UnitTypeId.MOTHERSHIP:
            if conceal:
                self.do(unit(AbilityId.MOTHERSHIPCLOAK_ORACLECLOAKFIELD))
            # No reliable cancel id for mothership field in LotV API — no-op on reveal.
            return
        if conceal:
            ability = self._BURROW_DOWN_BY_TYPE.get(tid)
        else:
            ability = self._BURROW_UP_BY_TYPE.get(tid)
        if ability is None:
            return
        self.do(unit(ability))

    async def apply_action(self, unit_tag: int, action):
        if not self.agent_system or "active_agent" not in self.agent_system.unit_agents:
            return

        unit = self.all_units.find_by_tag(unit_tag)
        if not unit:
            return

        if unit.type_id == UnitTypeId.SHIELDBATTERY:
            return

        arch = self.training_conf.get("agent_architecture", "original")

        agent = self.agent_system.unit_agents["active_agent"]
        # Spatial decode reads last_game_info; ensure it is set even if obs processing
        # failed earlier this step (otherwise every order targets world 0,0 → map corner).
        game_info = getattr(self, "game_info", None)
        if game_info is not None:
            agent.last_game_info = game_info

        if arch in ("universal", "squad"):
            # Universal/squad: action = [head_idx, spatial_idx] on 32×32 grid
            action = np.asarray(action, dtype=np.int64).reshape(-1)
            if action.shape[0] < 2:
                return
            head_idx = int(action[0])
            spatial_idx = int(action[1])

            action_heads = list(self.training_conf.get("action_heads") or [])
            categorical_heads = list(self.training_conf.get("categorical_heads") or [])
            n_spatial = len(action_heads)
            n_total = n_spatial + len(categorical_heads)
            if head_idx < 0 or head_idx >= n_total:
                return

            if head_idx >= n_spatial:
                head_name = categorical_heads[head_idx - n_spatial]
                if head_name in ("cloak_burrow", "uncloak_unburrow"):
                    conceal = head_name == "cloak_burrow"
                    self._mark_command_attempt(
                        unit, "cloak" if conceal else "uncloak"
                    )
                    self._issue_cloak_burrow(unit, conceal=conceal)
                    self._unit_last_order[unit_tag] = (head_name, None)
                    self.last_action_was_move_by_unit[unit_tag] = False
                return

            head_name = action_heads[head_idx]

            world_x, world_y = agent.convert_spatial_action_to_world(
                head_name, spatial_idx, active_unit=unit
            )
            target = Point2((world_x, world_y))

            if "heal" in head_name:
                self._mark_heal_command_attempt(unit)
                self._issue_heal_at_point(unit, target)
                self._unit_last_order[unit_tag] = (head_name, target)
                self.last_action_was_move_by_unit[unit_tag] = False
            elif self._refresh_heal_lock(unit):
                return
            elif "aoe" in head_name:
                issue_aoe_at_point(self, unit, target)
                self._unit_last_order[unit_tag] = (head_name, target)
                self.last_action_was_move_by_unit[unit_tag] = False
            elif "unload" in head_name:
                self._issue_unload_at_point(unit, target)
                self._unit_last_order[unit_tag] = (head_name, target)
                self.last_action_was_move_by_unit[unit_tag] = False
            elif "load" in head_name:
                self._issue_load_at_point(unit, target)
                self._unit_last_order[unit_tag] = (head_name, target)
                self.last_action_was_move_by_unit[unit_tag] = False
            elif "attack" in head_name:
                issue_attack_at_point(self, unit, target)
                self._mark_command_attempt(unit, "attack")
                self._unit_last_order[unit_tag] = (head_name, target)
                self.last_action_was_move_by_unit[unit_tag] = False
            else:
                self.do(unit.move(target))
                self._mark_command_attempt(unit, "move")
                self._unit_last_order[unit_tag] = (head_name, target)
                self.last_action_was_move_by_unit[unit_tag] = True
        else:
            # Original: flat integer action on grid_size×grid_size grid
            grid_area = self.grid_size * self.grid_size
            is_attack = False
            spatial_idx = action

            if action >= grid_area:
                is_attack = True
                spatial_idx = action - grid_area

            world_x, world_y = self.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world(spatial_idx)
            target = Point2((world_x, world_y))

            if is_attack:
                issue_attack_at_point(self, unit, target)
            else:
                self.do(unit.move(target))
            head_name_orig = "attack" if is_attack else "move"
            self._unit_last_order[unit_tag] = (head_name_orig, target)
            self.last_action_was_move_by_unit[unit_tag] = not is_attack

    def _enemies_for_unit(self, unit: Unit) -> List[Unit]:
        """Return hostile mobile units for ``unit`` in God-Mode (bot.player_id is always P1)."""
        return [u for u in _hostile_candidates(self, unit) if not u.is_structure]

    def _is_enemy_in_tactical_fovea(self, focal: Unit, enemy: Unit) -> bool:
        """True if ``enemy`` lies inside the 32×32 tactical crop centered on ``focal``."""
        if focal is None or enemy is None:
            return False
        h = float(self._TACTICAL_FOVEA_MAX_DELTA)
        dx = abs(float(enemy.position.x) - float(focal.position.x))
        dy = abs(float(enemy.position.y) - float(focal.position.y))
        return dx <= h and dy <= h

    def _enemy_com_in_tactical_fovea(
        self, focal: Unit, enemies: Iterable[Unit]
    ) -> Optional[Point2]:
        """Center of mass of hostile units visible in ``focal``'s tactical fovea."""
        in_fovea = [e for e in enemies if self._is_enemy_in_tactical_fovea(focal, e)]
        if not in_fovea:
            return None
        cx = sum(float(e.position.x) for e in in_fovea) / len(in_fovea)
        cy = sum(float(e.position.y) for e in in_fovea) / len(in_fovea)
        return Point2((cx, cy))

    @staticmethod
    def _dist_to_point(unit: Unit, target: Point2) -> float:
        return math.hypot(
            float(unit.position.x) - float(target.x),
            float(unit.position.y) - float(target.y),
        )

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

    def _is_ranged_combat_unit(self, unit: Unit, enemies: Iterable[Unit]) -> bool:
        """Cooldown-retreat shaping applies only to units that outrange melee contact."""
        # HellionTank (Hellbat) and Archon are effectively melee for cooldown-kite /
        # melee damage scaling even though their weapon range can exceed the generic
        # melee-vs-ranged threshold (1.5) in game/catalog data.
        if is_force_melee_unit(unit):
            return False
        enemy_list = [e for e in enemies if e is not None]
        if enemy_list:
            max_range = max(
                self._weapon_range_attacker_vs_defender(unit, e) for e in enemy_list
            )
        else:
            gr = float(getattr(unit, "ground_range", 0.0) or 0.0)
            ar = float(getattr(unit, "air_range", 0.0) or 0.0)
            max_range = max(gr, ar)
        return max_range > self._RANGED_WEAPON_RANGE_MIN

    def _is_melee_combat_unit(self, unit: Unit, enemies: Iterable[Unit]) -> bool:
        """Melee contact units (inverse of ranged gate used for cooldown retreat)."""
        return not self._is_ranged_combat_unit(unit, enemies)

    def _squad_has_melee_unit(self, squad_tags: List[int], enemies: Iterable[Unit]) -> bool:
        for tag in squad_tags:
            unit = self.all_units.find_by_tag(tag)
            if unit is not None and self._is_melee_combat_unit(unit, enemies):
                return True
        return False

    @staticmethod
    def _scaled_damage_reward_coeff(r_conf: Dict, *, squad_is_melee: bool) -> float:
        base = float(r_conf.get("damage_reward", 0.1))
        if not squad_is_melee:
            return base
        melee_mult = float(r_conf.get("melee_multiplier", 0.5))
        return base * (melee_mult + 1.0)

    @staticmethod
    def _scaled_damage_penalty_coeff(r_conf: Dict, *, squad_is_melee: bool) -> float:
        base = float(r_conf.get("damage_penalty", 0.1))
        if not squad_is_melee:
            return base
        melee_mult = float(r_conf.get("melee_multiplier", 0.5))
        return base * melee_mult

    def _defend_pylon_pos_for_unit(
        self, unit: Unit, fallback_pylon_pos: Optional[Point2] = None
    ) -> Optional[Point2]:
        """Own pylon anchor for a defender unit (spawn-time target preferred)."""
        if unit.tag in self.unit_target_map:
            return self.unit_target_map[unit.tag]
        if fallback_pylon_pos is not None:
            return fallback_pylon_pos
        owner = int(getattr(unit, "owner_id", 0) or 0)
        if owner in (1, 2):
            allies = allied_candidates_for_owner(self, owner)
            pylon = first_unit_of_type(allies, UnitTypeId.PYLON)
            return pylon.position if pylon is not None else None
        if unit.owner_id == self.player_id:
            structs = self.structures
        else:
            structs = self.enemy_structures
        pylon = first_unit_of_type(structs, UnitTypeId.PYLON)
        return pylon.position if pylon is not None else None

    def record_pre_action_retreat_state(self, unit_tag: int) -> None:
        """
        Snapshot pre-action state for per-step distance shaping.

        Captures weapon cooldown / enemy-CoM distance (cooldown retreat) and
        distance to the defender pylon (defender zone penalty).
        """
        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
            return

        self._pre_action_weapon_cd_by_unit[unit_tag] = float(
            getattr(unit, "weapon_cooldown", 0.0) or 0.0
        )

        enemies = self._enemies_for_unit(unit)
        enemy_com = self._enemy_com_in_tactical_fovea(unit, enemies)
        if enemy_com is None:
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
        else:
            self._pre_action_dist_to_enemy_com[unit_tag] = self._dist_to_point(unit, enemy_com)

        if self.unit_command_map.get(unit_tag, "attack") == "defend":
            pylon_pos = self._defend_pylon_pos_for_unit(unit)
            if pylon_pos is not None:
                self._pre_action_dist_to_defend_pylon[unit_tag] = self._dist_to_point(
                    unit, pylon_pos
                )
            else:
                self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
        else:
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)

    def calculate_cooldown_retreat_reward(
        self, unit_tag: int, enemies: Iterable[Unit]
    ) -> float:
        """
        Reward moving away from the enemy cluster CoM while the weapon is on cooldown.

        Gates (all required):
        - ``cooldown_retreat_scale`` > 0 in config
        - Unit has ranged weapon range (> ``_RANGED_WEAPON_RANGE_MIN``)
        - Pre-action weapon cooldown > 0
        - Last issued command was a move (not attack)
        - At least one enemy in the unit's tactical fovea (pre- and post-step)
        - Distance to enemy CoM increased since the pre-action snapshot
        """
        if unit_tag is None:
            return 0.0

        scale = float(
            self.training_conf.get("rewards", {}).get("cooldown_retreat_scale", 0.0)
        )
        if scale == 0.0:
            return 0.0

        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            return 0.0

        if not self._is_ranged_combat_unit(unit, enemies):
            return 0.0

        pre_cd = self._pre_action_weapon_cd_by_unit.get(unit_tag)
        prev_dist = self._pre_action_dist_to_enemy_com.get(unit_tag)
        if pre_cd is None or prev_dist is None:
            return 0.0

        eps = 1e-3
        if float(pre_cd) <= eps:
            return 0.0

        if not self.last_action_was_move_by_unit.get(unit_tag, False):
            return 0.0

        enemy_com = self._enemy_com_in_tactical_fovea(unit, enemies)
        if enemy_com is None:
            return 0.0

        curr_dist = self._dist_to_point(unit, enemy_com)
        if curr_dist <= prev_dist + eps:
            return 0.0

        return scale * (curr_dist - prev_dist)

    def calculate_defender_zone_penalty(
        self, unit_tag: int, own_pylon_pos: Optional[Point2]
    ) -> float:
        """
        Penalize defender units that move farther from their pylon while outside
        the defender zone (distance > ``defender_zone_radius`` map units).

        No positive shaping — defenders are only penalized for stepping away
        beyond the zone until they return inside it.
        """
        if unit_tag is None:
            return 0.0

        if self.unit_command_map.get(unit_tag, "attack") != "defend":
            return 0.0

        r_conf = self.training_conf.get("rewards", {})
        dist_mult = float(r_conf.get("distance_multiplier", 0.1))
        zone_radius = float(r_conf.get("defender_zone_radius", 10.0))
        if dist_mult == 0.0:
            return 0.0

        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
            return 0.0

        pylon_pos = self._defend_pylon_pos_for_unit(unit, own_pylon_pos)
        if pylon_pos is None:
            return 0.0

        prev_dist = self._pre_action_dist_to_defend_pylon.get(unit_tag)
        if prev_dist is None:
            return 0.0

        curr_dist = self._dist_to_point(unit, pylon_pos)
        eps = 1e-3

        # Inside the defender zone — no penalty from this signal.
        if curr_dist <= zone_radius + eps:
            return 0.0

        # Outside the zone: penalize only when moving farther from the pylon.
        if curr_dist > prev_dist + eps:
            return -dist_mult * (curr_dist - prev_dist)

        return 0.0

    @staticmethod
    def _iter_combat_units_with_passengers(
        units: Iterable,
        *,
        include_filter=None,
    ) -> List:
        """Visible non-structure mobiles plus embarked passengers (deduped by tag)."""
        seen = set()
        out: List = []
        for u in units or []:
            if u is None:
                continue
            if include_filter is not None:
                if not include_filter(u):
                    continue
            elif getattr(u, "is_structure", False):
                continue
            tag = getattr(u, "tag", None)
            if tag is not None and tag not in seen:
                seen.add(tag)
                out.append(u)
            for p in getattr(u, "passengers", None) or []:
                if p is None:
                    continue
                if include_filter is not None and not include_filter(p):
                    continue
                ptag = getattr(p, "tag", None)
                if ptag is not None and ptag not in seen:
                    seen.add(ptag)
                    out.append(p)
        return out

    @classmethod
    def _mobile_combat_count(cls, units: Iterable, *, include_filter=None) -> int:
        return len(cls._iter_combat_units_with_passengers(units, include_filter=include_filter))

    @classmethod
    def _mobile_combat_hp_sum(cls, units: Iterable, *, include_filter=None) -> float:
        total = 0.0
        for u in cls._iter_combat_units_with_passengers(units, include_filter=include_filter):
            total += float(getattr(u, "health", 0.0) or 0.0) + float(
                getattr(u, "shield", 0.0) or 0.0
            )
        return total

    @classmethod
    def _combat_snapshot(
        cls, units: Iterable, *, include_filter=None
    ) -> Dict[int, Tuple[float, Optional[Point2]]]:
        """Per-tag HP and world position, using the transport position for cargo."""
        snap: Dict[int, Tuple[float, Optional[Point2]]] = {}
        seen = set()
        for u in units or []:
            if u is None:
                continue
            if include_filter is not None:
                if not include_filter(u):
                    continue
            elif getattr(u, "is_structure", False):
                continue
            carrier_pos = getattr(u, "position", None)
            tag = getattr(u, "tag", None)
            if tag is not None and tag not in seen:
                seen.add(tag)
                hp = float(getattr(u, "health", 0.0) or 0.0) + float(
                    getattr(u, "shield", 0.0) or 0.0
                )
                snap[tag] = (hp, carrier_pos)
            for p in getattr(u, "passengers", None) or []:
                if p is None:
                    continue
                if include_filter is not None and not include_filter(p):
                    continue
                ptag = getattr(p, "tag", None)
                if ptag is not None and ptag not in seen:
                    seen.add(ptag)
                    php = float(getattr(p, "health", 0.0) or 0.0) + float(
                        getattr(p, "shield", 0.0) or 0.0
                    )
                    snap[ptag] = (php, carrier_pos)
        return snap

    @staticmethod
    def _is_in_defender_zone(
        pos: Optional[Point2], pylon_pos: Optional[Point2], zone_radius: float
    ) -> bool:
        if pos is None or pylon_pos is None:
            return False
        return math.hypot(
            float(pos.x) - float(pylon_pos.x),
            float(pos.y) - float(pylon_pos.y),
        ) <= zone_radius + 1e-3

    def _defender_zone_anchor(
        self, squad_tags: List[int], fallback_pylon_pos: Optional[Point2]
    ) -> Optional[Point2]:
        for tag in squad_tags:
            if tag in self.unit_target_map:
                return self.unit_target_map[tag]
        return fallback_pylon_pos

    def _defender_zone_credited_enemy_combat(
        self,
        p_id: int,
        curr_snap: Dict[int, Tuple[float, Optional[Point2]]],
        pylon_pos: Optional[Point2],
        zone_radius: float,
    ) -> Tuple[float, int]:
        """HP lost and kills among enemies last seen inside the defender zone."""
        prev_snap = self.prev_enemy_combat_snapshot.get(p_id) or {}
        hp_delta = 0.0
        kills = 0
        for tag, (prev_hp, prev_pos) in prev_snap.items():
            if tag in curr_snap:
                curr_hp, curr_pos = curr_snap[tag]
                lost = prev_hp - curr_hp
                if lost <= 0:
                    continue
                pos = curr_pos if curr_pos is not None else prev_pos
                if self._is_in_defender_zone(pos, pylon_pos, zone_radius):
                    hp_delta += lost
            elif self._is_in_defender_zone(prev_pos, pylon_pos, zone_radius):
                if prev_hp > 0:
                    hp_delta += prev_hp
                kills += 1
        return hp_delta, kills

    @staticmethod
    def _squad_is_defend(squad_tags: List[int], unit_command_map: Dict[int, str]) -> bool:
        return bool(squad_tags) and all(
            unit_command_map.get(tag, "attack") == "defend" for tag in squad_tags
        )

    def _alliance_perspective_groups(self, p_id: int) -> Tuple[Any, Any, Any, Any]:
        """``(own_units, enemy_units, own_structs, enemy_structs)`` from alliance buckets."""
        return alliance_perspective_groups(self, p_id)

    def _perspective_unit_groups(self, p_id: int) -> Tuple[Any, Any, Any, Any]:
        """Own/enemy mobiles and structures for *p_id* (owner-id, then alliance)."""
        return perspective_unit_groups(self, p_id)

    @staticmethod
    def _first_pylon(structs: Any) -> Optional[Any]:
        """First pylon in a Units collection, list, or owner-filtered group."""
        return first_unit_of_type(structs, UnitTypeId.PYLON)

    def calculate_squad_step_reward(
        self,
        squad_tags: List[int],
        perspective_player_id: Optional[int] = None,
    ) -> float:
        """
        Reward for one squad physics step.

        Global signals (pylon/army HP, attacker CoM) are applied once.
        Per-unit shaping (cooldown retreat, defender zone, step penalty) is summed
        over ``squad_tags``.

        Defend-role squads only receive ``damage_reward`` / ``kill_reward`` for
        enemies inside ``defender_zone_radius``. Own ``damage_penalty`` /
        ``kill_penalty`` still apply when defenders are outside the zone.
        """
        self._step_cooldown_retreat_reward = 0.0
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id

        reward = 0.0
        r_conf = self.training_conf.get("rewards", {})

        own_units, enemy_units, own_structs, enemy_structs = self._perspective_unit_groups(
            p_id
        )
        own_pylon = self._first_pylon(own_structs)
        enemy_pylon = self._first_pylon(enemy_structs)
        if own_pylon is None or enemy_pylon is None:
            _, _, bucket_own_s, bucket_en_s = self._alliance_perspective_groups(p_id)
            if own_pylon is None:
                own_pylon = self._first_pylon(bucket_own_s)
            if enemy_pylon is None:
                enemy_pylon = self._first_pylon(bucket_en_s)

        # Update persistent position if currently visible
        if enemy_pylon:
            self.enemy_pylon_pos[p_id] = enemy_pylon.position

        target_pos = self.enemy_pylon_pos.get(p_id)

        # 1. Pylon Damage Rewards (Terminal/Heavy signal)
        def get_enemy_hp_safe(pylon_unit, last_hp, target_position):
            if pylon_unit:
                return pylon_unit.health + pylon_unit.shield
            
            # If unit is missing, check visibility of its position
            if target_position and self.is_visible(target_position):
                # We see the position, but no unit -> IT IS DEAD
                return 0.0
            
            # Not visible -> Assume unchanged (Fog of War)
            return last_hp

        def get_own_hp(pylon_unit):
            # We always have info on our own units
            if not pylon_unit: return 0.0
            return pylon_unit.health + pylon_unit.shield

        own_pylon_pos = own_pylon.position if own_pylon is not None else None
        
        own_pylon_hp = get_own_hp(own_pylon)
        
        own_global_idx = p_id
        enemy_global_idx = 2 if p_id == 1 else 1
        
        enemy_pylon_hp = get_enemy_hp_safe(enemy_pylon, self.prev_pylon_health[enemy_global_idx], target_pos)
        
        # Track global pylon health by persistent player ID (1 or 2)
        p1_pylon_hp = own_pylon_hp if p_id == 1 else enemy_pylon_hp
        p2_pylon_hp = enemy_pylon_hp if p_id == 1 else own_pylon_hp
        
        curr_pylon_hp_map = {1: p1_pylon_hp, 2: p2_pylon_hp}
        
        # Pylon Damage Rewards (Signal for acting player)
        if curr_pylon_hp_map[enemy_global_idx] < self.prev_pylon_health[enemy_global_idx]:
            reward += (self.prev_pylon_health[enemy_global_idx] - curr_pylon_hp_map[enemy_global_idx]) * r_conf.get("pylon_damage_reward", 2.0)
            
        if curr_pylon_hp_map[own_global_idx] < self.prev_pylon_health[own_global_idx]:
            reward -= (self.prev_pylon_health[own_global_idx] - curr_pylon_hp_map[own_global_idx]) * r_conf.get("pylon_damage_penalty", 2.0)

        # Update global health tracking
        self.prev_pylon_health[1] = p1_pylon_hp
        self.prev_pylon_health[2] = p2_pylon_hp
        
        # 2. Combat Rewards (Mobile Units)
        mobile_enemies_for_melee = [u for u in enemy_units if not u.is_structure]
        squad_is_melee = self._squad_has_melee_unit(squad_tags, mobile_enemies_for_melee)
        damage_reward_coeff = self._scaled_damage_reward_coeff(
            r_conf, squad_is_melee=squad_is_melee
        )
        damage_penalty_coeff = self._scaled_damage_penalty_coeff(
            r_conf, squad_is_melee=squad_is_melee
        )

        # Permanent-army HP only (skip timed summons' bodies). Shots from
        # Interceptors still count: they change permanent units' HP, which is
        # what this ledger tracks. We only ignore Interceptor launch/recall/
        # despawn as fake damage_reward / kill_reward.
        curr_enemy_hp = self._mobile_combat_hp_sum(
            enemy_units, include_filter=counts_for_combat_hp_reward
        )
        curr_enemy_count = self._mobile_combat_count(
            enemy_units, include_filter=counts_for_combat_hp_reward
        )
        curr_enemy_snap = self._combat_snapshot(
            enemy_units, include_filter=counts_for_combat_hp_reward
        )
        kill_reward_val = float(r_conf.get("kill_reward", 0.0))
        kill_penalty_val = float(r_conf.get("kill_penalty", 0.0))
        squad_is_defend = self._squad_is_defend(squad_tags, self.unit_command_map)

        # Defend role: only credit enemy damage/kills inside defender_zone_radius.
        # Attack role keeps the global HP/count delta. Own attrition is never gated.
        if squad_is_defend:
            zone_radius = float(r_conf.get("defender_zone_radius", 10.0))
            zone_anchor = self._defender_zone_anchor(squad_tags, own_pylon_pos)
            zoned_hp_delta, zoned_kills = self._defender_zone_credited_enemy_combat(
                p_id, curr_enemy_snap, zone_anchor, zone_radius
            )
            if zoned_hp_delta > 0:
                reward += zoned_hp_delta * damage_reward_coeff
            if kill_reward_val != 0.0 and zoned_kills > 0:
                reward += kill_reward_val * zoned_kills
        else:
            if curr_enemy_hp < self.prev_enemy_units_health[p_id]:
                reward += (self.prev_enemy_units_health[p_id] - curr_enemy_hp) * damage_reward_coeff
            if kill_reward_val != 0.0 and curr_enemy_count < self.prev_enemy_unit_count[p_id]:
                reward += kill_reward_val * (self.prev_enemy_unit_count[p_id] - curr_enemy_count)

        self.prev_enemy_units_health[p_id] = curr_enemy_hp
        self.prev_enemy_unit_count[p_id] = curr_enemy_count
        self.prev_enemy_combat_snapshot[p_id] = curr_enemy_snap

        curr_own_hp = self._mobile_combat_hp_sum(
            own_units, include_filter=counts_for_combat_hp_reward
        )
        prev_own_hp = self.prev_own_units_health[p_id]
        heal_coeff = float(r_conf.get("heal_reward", 0.0))
        # prev_own == 0 after reset_episode_state is "no baseline yet", not an
        # empty army. Crediting that delta as heal pays ~0.1 × full army HP on
        # the first step of every episode and blows up return scale / EV.
        own_baseline_ready = (
            float(self.prev_own_unit_count.get(p_id, 0) or 0) > 0 or prev_own_hp > 0.0
        )
        # cast_tags: fresh-cast-only, gates heal_cast_reward (see
        # _issue_heal_at_point's continuing_same_patient comment).
        # order_tags: fresh OR continuing, gates heal_reward's per-HP shaping
        # so a multi-step Medivac beam / SCV repair keeps earning HP credit
        # on every step it actually ticks HP, not just the step that started
        # the lock (which usually has not ticked any HP yet).
        n_issued = 0
        cast_tags = getattr(self, "_heal_cast_tags", None)
        if isinstance(cast_tags, set) and squad_tags:
            n_issued = sum(1 for tag in squad_tags if int(tag) in cast_tags)
        cast_bonus = float(r_conf.get("heal_cast_reward", 0.0))
        if n_issued > 0 and cast_bonus != 0.0:
            reward += cast_bonus * n_issued
        order_tags = getattr(self, "_heal_order_tags", None)
        healing_active = (
            isinstance(order_tags, set)
            and bool(squad_tags)
            and any(int(tag) in order_tags for tag in squad_tags)
        )
        if (
            heal_coeff != 0.0
            and healing_active
            and own_baseline_ready
            and curr_own_hp > prev_own_hp
        ):
            reward += (curr_own_hp - prev_own_hp) * heal_coeff
        if isinstance(cast_tags, set):
            cast_tags.clear()
        if isinstance(order_tags, set):
            order_tags.clear()
        attempt_tags = getattr(self, "_heal_attempt_tags", None)
        if isinstance(attempt_tags, set):
            attempt_tags.clear()
        self._clear_command_attempt_tags()
        if own_baseline_ready and curr_own_hp < prev_own_hp:
            reward -= (prev_own_hp - curr_own_hp) * damage_penalty_coeff
        self.prev_own_units_health[p_id] = curr_own_hp

        # Kill bonuses: flat per-unit reward/penalty on top of continuous HP shaping.
        # Counts include embarked passengers so transport deaths credit cargo and
        # load/unload alone does not look like a kill.
        curr_own_count = self._mobile_combat_count(
            own_units, include_filter=counts_for_combat_hp_reward
        )
        if kill_penalty_val != 0.0 and curr_own_count < self.prev_own_unit_count[p_id]:
            reward += kill_penalty_val * (self.prev_own_unit_count[p_id] - curr_own_count)
        self.prev_own_unit_count[p_id] = curr_own_count

        # 3. Progress Rewards — attacker squad Center-of-Mass distance shaping
        #
        # When an attacker unit is active: reward the CoM of ALL attacker units
        # (same player) closing in on the target pylon.
        # Defenders use per-unit zone penalty (section 3c) instead of CoM shaping.
        active_unit_role = self.unit_command_map.get(self.active_unit_tag, "attack") if self.active_unit_tag else "attack"

        if active_unit_role == "attack":
            squad_units = [u for u in own_units if not u.is_structure
                           and self.unit_command_map.get(u.tag, "attack") == "attack"]
            squad_target = next(
                (self.unit_target_map[u.tag] for u in squad_units if u.tag in self.unit_target_map),
                target_pos,
            )
            dist_key = (p_id, active_unit_role)
            dist_mult = r_conf.get("distance_multiplier", 0.1)

            if squad_units and squad_target is not None:
                cx = sum(u.position.x for u in squad_units) / len(squad_units)
                cy = sum(u.position.y for u in squad_units) / len(squad_units)
                curr_com_dist = ((cx - squad_target.x) ** 2 + (cy - squad_target.y) ** 2) ** 0.5
                prev_com_dist = self.prev_dist_per_squad.get(dist_key)
                if prev_com_dist is not None:
                    reward += (prev_com_dist - curr_com_dist) * dist_mult
                self.prev_dist_per_squad[dist_key] = curr_com_dist
            else:
                self.prev_dist_per_squad.pop(dist_key, None)
        else:
            self.prev_dist_per_squad.pop((p_id, "defend"), None)

        # 3b–4. Per-unit cooldown/zone shaping; step penalty once per physics step.
        mobile_enemies = mobile_enemies_for_melee
        saved_active = self.active_unit_tag
        squad_is_attack = (
            any(self.unit_command_map.get(tag, "attack") == "attack" for tag in squad_tags)
            if squad_tags
            else True
        )
        for tag in squad_tags:
            self.active_unit_tag = tag
            retreat = self.calculate_cooldown_retreat_reward(tag, mobile_enemies)
            self._step_cooldown_retreat_reward = (
                float(getattr(self, "_step_cooldown_retreat_reward", 0.0)) + float(retreat)
            )
            reward += retreat
            reward += self.calculate_defender_zone_penalty(tag, own_pylon_pos)
        self.active_unit_tag = saved_active

        if squad_tags:
            if squad_is_attack:
                reward += r_conf.get("step_penalty", -0.05)
            else:
                reward += r_conf.get("defender_step_reward", 0.001)

        return reward

    def calculate_reward(self, perspective_player_id: Optional[int] = None) -> float:
        """Single-unit wrapper around :meth:`calculate_squad_step_reward`."""
        squad_tags = [self.active_unit_tag] if self.active_unit_tag is not None else []
        return self.calculate_squad_step_reward(squad_tags, perspective_player_id)
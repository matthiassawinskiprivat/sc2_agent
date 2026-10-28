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
    AOE_CAST_ANIMATION_SECONDS,
    AOE_CAST_RANGE,
    AOE_SPLASH_RADIUS,
    counts_for_combat_hp_reward,
    first_unit,
    first_unit_of_type,
    heal_click_cell_half,
    heal_click_hits_unit,
    heal_target_is_eligible,
    HEAL_CAST_RANGE,
    is_force_melee_unit,
    issue_aoe_at_point,
    issue_attack_at_point,
    issue_snipe_on_unit,
    perspective_unit_groups,
    neural_command_allowed,
    resolve_air_hostile_at_point,
    SNIPE_CAST_ANIMATION_SECONDS,
    SNIPE_CAST_RANGE,
    _AOE_ENERGY_COST,
    _SNIPE_ENERGY_COST,
    unit_in_tactical_fovea,
    unit_is_neuraled,
    uses_caustic_spray_on_structures,
    uses_purification_nova_for_attack,
    uses_spawn_locusts_for_attack,
)
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.squad_constants import (
    ALLY_TARGET_COMMAND_HEADS,
    ENEMY_TARGET_COMMAND_HEADS,
    LOAD_TARGET_COMMAND_HEADS,
    SNIPE_COMMAND_HEADS,
)
from src.agent.ctf_economic_rewards import (
    ROLE_ATTACK,
    ROLE_DEFEND,
    EconomicRewardConfig,
    UnitValue,
    build_value_snapshot,
    apply_neural_reward_buckets,
    objective_credit_scale,
    pylon_fraction_credit,
    pylon_max_vitality,
    retained_value_fraction,
    snapshot_total_value,
    unit_economic_value,
    value_delta,
)

# Per-step economic telemetry keys (all in normalised economic value units).
ECONOMIC_METRIC_KEYS: Tuple[str, ...] = (
    "economic_damage_dealt",
    "economic_enemy_killed",
    "economic_attacker_loss",
    "economic_defender_loss",
    "economic_healing",
    "aoe_land_value",
    "detection_reward",
    "defender_zone_shaping",
    "snipe_target_value",
    "spellcaster_retreat",
    "conceal_shaping",
)

# Ghost, High Templar, Infestor, and Viper cast from behind the army. They are
# left out of attacker center-of-mass distance shaping so that reward does not
# drag them into the objective. Burrowed Infestors stay in the set.
_BACKLINE_SPELLCASTER_TYPES: frozenset = frozenset(
    {
        UnitTypeId.GHOST,
        UnitTypeId.HIGHTEMPLAR,
        UnitTypeId.INFESTOR,
        UnitTypeId.INFESTORBURROWED,
        UnitTypeId.VIPER,
    }
)
# Swarm Hosts launch locusts and then wait. While that launch is on cooldown
# they are treated like a caster with no energy: left out of the attacker
# center of mass, and paid to open space from the nearest enemy. A ready host
# stays in the center of mass so it can walk into launch range.
_SWARM_HOST_TYPES: frozenset = frozenset(
    {
        UnitTypeId.SWARMHOSTMP,
        UnitTypeId.SWARMHOSTBURROWEDMP,
    }
)

# Planted ambushers. Burrow credit scales with how many enemies sit inside the
# burrowed attack range. Lurker spines hit ground only (range 8). Widow Mine
# sentinel missiles hit ground and air (range 5). A reported weapon range above
# the base (Seismic Spines, for example) wins when the unit exposes one.
_LURKER_AMBUSH_TYPES: frozenset = frozenset(
    {
        UnitTypeId.LURKERMP,
        UnitTypeId.LURKERMPBURROWED,
        UnitTypeId.LURKERBURROWED,
    }
)
_AMBUSH_ATTACK_RANGE: Dict[UnitTypeId, float] = {
    UnitTypeId.LURKERMP: 8.0,
    UnitTypeId.LURKERMPBURROWED: 8.0,
    UnitTypeId.LURKERBURROWED: 8.0,
    UnitTypeId.WIDOWMINE: 5.0,
    UnitTypeId.WIDOWMINEBURROWED: 5.0,
}
_AMBUSH_BURROWED_TYPES: frozenset = frozenset(
    {
        UnitTypeId.LURKERMPBURROWED,
        UnitTypeId.LURKERBURROWED,
        UnitTypeId.WIDOWMINEBURROWED,
    }
)
# Health fraction at or below this counts as "about to die" for cloak/burrow.
_LOW_HP_CONCEAL_FRACTION = 0.10
# Low-energy casters are paid for opening space only until this gap.
_SPELLCASTER_RETREAT_DISTANCE = 10.0
EMPTY_ECONOMIC_METRICS: Dict[str, float] = {key: 0.0 for key in ECONOMIC_METRIC_KEYS}

# Command heads whose terminal target is a unit pointer, not a world cell.
UNIT_TARGET_COMMAND_HEADS: Tuple[str, ...] = (
    ALLY_TARGET_COMMAND_HEADS
    + LOAD_TARGET_COMMAND_HEADS
    + ENEMY_TARGET_COMMAND_HEADS
    + SNIPE_COMMAND_HEADS
)

# Per-step TensorBoard command mix (trainee tags only; see env _record_command_metrics).
COMMAND_METRIC_KINDS: Tuple[str, ...] = (
    "move",
    "attack",
    "cloak",
    "uncloak",
    "aoe",
    "snipe",
    "load",
    "unload",
)


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
    _DEFAULT_SNIPE_CASTER_MOVE_SPEED: Dict[UnitTypeId, float] = {
        UnitTypeId.GHOST: 3.15,
        UnitTypeId.BATTLECRUISER: 2.62,
        UnitTypeId.INFESTOR: 2.95,
        UnitTypeId.INFESTORBURROWED: 0.0,
        UnitTypeId.HIGHTEMPLAR: 2.62,
    }
    _DEFAULT_SNIPE_CAST_ANIMATION_SECONDS: float = 1.0
    _DEFAULT_AOE_CASTER_MOVE_SPEED: Dict[UnitTypeId, float] = {
        UnitTypeId.INFESTOR: 2.95,
        UnitTypeId.INFESTORBURROWED: 0.0,
        UnitTypeId.HIGHTEMPLAR: 2.62,
        UnitTypeId.GHOST: 3.15,
        UnitTypeId.RAVAGER: 3.85,
        UnitTypeId.VIPER: 4.13,
    }
    _DEFAULT_AOE_CAST_ANIMATION_SECONDS: float = 0.5

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
        # tag → UnitValue (economic value, vitality, position, role) per
        # perspective. Drives the resource-weighted role rewards; the HP ledgers
        # above remain only as the baseline-ready / fog-of-war bookkeeping they
        # always were. See src/agent/ctf_economic_rewards.py.
        self.prev_enemy_value_snapshot: Dict[int, Dict[int, UnitValue]] = {1: {}, 2: {}}
        self.prev_own_value_snapshot: Dict[int, Dict[int, UnitValue]] = {1: {}, 2: {}}
        # Frozen at ``prime_combat_reward_baselines``: owner → spawn army value /
        # pylon max vitality. Pylon damage pays fraction × scale × army value.
        self._episode_army_value: Dict[int, float] = {1: 0.0, 2: 0.0}
        self._episode_pylon_max_vitality: Dict[int, float] = {1: 0.0, 2: 0.0}
        # Per-step economic telemetry, popped by the env for TensorBoard.
        self._step_economic_metrics: Dict[str, float] = dict(EMPTY_ECONOMIC_METRICS)
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
        # Enemy CoM is frozen at pre-action so opponent chase in the same physics
        # step cannot cancel credit for the unit's own retreat (Marine vs Zergling).
        self._pre_action_weapon_cd_by_unit: Dict[int, float] = {}
        self._pre_action_dist_to_enemy_com: Dict[int, float] = {}
        self._pre_action_enemy_com: Dict[int, Point2] = {}
        self._pre_action_dist_to_defend_pylon: Dict[int, float] = {}
        self._pre_action_vitality: Dict[int, float] = {}
        # Nearest hostile point, frozen before the order, for low-energy caster
        # spacing. Distance is to that frozen point so a chase cannot erase it.
        self._pre_action_nearest_enemy: Dict[int, Point2] = {}
        self._pre_action_dist_to_nearest_enemy: Dict[int, float] = {}
        self._pre_action_spell_energy: Dict[int, float] = {}
        self._step_aoe_land_value: float = 0.0
        # Economic value of enemies a fresh ``snipe`` was aimed at this step.
        # Each enemy tag is credited at most once per episode.
        self._step_snipe_target_value: float = 0.0
        self._snipe_value_paid_tags: Set[int] = set()
        # Low-HP conceal / ambush burrow payouts accrued at command issue.
        self._step_conceal_shaping: float = 0.0
        # Tags that already collected the empty-field unburrow bonus and have
        # not since planted on an enemy. Stops burrow/unburrow toggling in the
        # open from farming the bonus every other step.
        self._ambush_unburrow_disarmed: Set[int] = set()
        # True when this unit's last *policy* order was a move / attack. Attack is
        # snapshotted at record_pre_action time as "fired last step" so the kite
        # window still opens when weapon_cooldown has already ticked to 0 across
        # a long multi-unit physics step (Marine CD ≈ one paired squad step).
        self.last_action_was_move_by_unit: Dict[int, bool] = {}
        self.last_action_was_attack_by_unit: Dict[int, bool] = {}
        self._pre_action_attacked_prev_step: Dict[int, bool] = {}
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
        # Selected tags that received move / attack / cloak / uncloak / aoe /
        # load / unload this step.
        self._command_attempt_tags: Dict[str, Set[int]] = {
            kind: set() for kind in COMMAND_METRIC_KINDS
        }
        # Sum of cooldown-retreat shaping paid this physics step (TensorBoard).
        self._step_cooldown_retreat_reward: float = 0.0
        # healer_tag → (patient_tag, expiry_game_loop) while walk-in + cast
        # animation must not be overwritten by attack/move.
        self._heal_lock: Dict[int, Tuple[int, int]] = {}
        # caster_tag → (target_tag, expiry_game_loop) while Yamato / Steady
        # Targeting / Neural / Feedback walk-in + prep must not be cancelled.
        self._snipe_lock: Dict[int, Tuple[int, int]] = {}
        # caster_tag → (point_or_air_tag, expiry_game_loop) while bile / storm /
        # fungal / EMP / parasitic-bomb walk-in + cast must not be cancelled.
        self._aoe_lock: Dict[int, Tuple[Any, int]] = {}
        # caster_tag → game_loop when a locally stamped spell is ready again.
        self._heal_ability_ready_after_loop: Dict[int, int] = {}
        self._aoe_ability_ready_after_loop: Dict[int, int] = {}
        self._snipe_ability_ready_after_loop: Dict[int, int] = {}
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
        """How many selected units were given this command kind this step."""
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
        """Record that ``unit`` received this command kind this step."""
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
        self.prev_enemy_value_snapshot = {1: {}, 2: {}}
        self.prev_own_value_snapshot = {1: {}, 2: {}}
        self._episode_army_value = {1: 0.0, 2: 0.0}
        self._episode_pylon_max_vitality = {1: 0.0, 2: 0.0}
        self._step_economic_metrics = dict(EMPTY_ECONOMIC_METRICS)
        self.prev_dist_to_pylon = {1: None, 2: None}
        self.prev_dist_per_unit = {}
        self.unit_command_map = {}
        self.unit_target_map = {}
        self._unit_last_order = {}
        self.prev_dist_per_squad = {}
        self._pre_action_weapon_cd_by_unit = {}
        self._pre_action_dist_to_enemy_com = {}
        self._pre_action_enemy_com = {}
        self._pre_action_dist_to_defend_pylon = {}
        self._pre_action_vitality = {}
        self._step_aoe_land_value = 0.0
        self._step_snipe_target_value = 0.0
        self._snipe_value_paid_tags = set()
        self._step_conceal_shaping = 0.0
        self._ambush_unburrow_disarmed = set()
        self._pre_action_nearest_enemy = {}
        self._pre_action_dist_to_nearest_enemy = {}
        self._pre_action_spell_energy = {}
        self._detection_paid_tags = set()
        self.last_action_was_move_by_unit = {}
        self.last_action_was_attack_by_unit = {}
        self._pre_action_attacked_prev_step = {}
        self._heal_cast_tags = set()
        self._heal_order_tags = set()
        self._heal_attempt_tags = set()
        self._command_attempt_tags = {kind: set() for kind in COMMAND_METRIC_KINDS}
        self._step_cooldown_retreat_reward = 0.0
        self._heal_lock = {}
        self._snipe_lock = {}
        self._aoe_lock = {}
        self._heal_ability_ready_after_loop = {}
        self._aoe_ability_ready_after_loop = {}
        self._snipe_ability_ready_after_loop = {}
        self._available_abilities_by_tag = {}
        self.cached_observation = None
        self.frame_count = 0

    def pop_cooldown_retreat_reward(self) -> float:
        """Return and clear the cooldown-retreat shaping paid this physics step."""
        total = float(getattr(self, "_step_cooldown_retreat_reward", 0.0) or 0.0)
        self._step_cooldown_retreat_reward = 0.0
        return total

    def prime_combat_reward_baselines(self) -> None:
        """Snapshot army value/HP/pylon HP after spawn.

        Without this the first physics step compares a full army against an empty
        baseline and pays the whole army's economic value as healing.
        """
        econ = EconomicRewardConfig.from_rewards_conf(
            self.training_conf.get("rewards", {})
        )
        self._detection_paid_tags = set()
        for p_id in (1, 2):
            own_units, enemy_units, own_structs, enemy_structs = self._perspective_unit_groups(
                p_id
            )
            own_units, enemy_units, _suspended = apply_neural_reward_buckets(
                own_units, enemy_units, p_id
            )
            self.prev_own_value_snapshot[p_id] = build_value_snapshot(
                own_units,
                config=econ,
                include_filter=counts_for_combat_hp_reward,
                role_for_tag=lambda tag: self.unit_command_map.get(tag, ROLE_ATTACK),
            )
            self.prev_enemy_value_snapshot[p_id] = build_value_snapshot(
                enemy_units,
                config=econ,
                include_filter=counts_for_combat_hp_reward,
            )
            # Freeze spawn army value by SC2 owner so pylon credit stays proportional
            # to the army that was fielded, not the remnants after the fight starts.
            enemy_owner = 2 if p_id == 1 else 1
            self._episode_army_value[p_id] = snapshot_total_value(
                self.prev_own_value_snapshot[p_id]
            )
            self._episode_army_value[enemy_owner] = snapshot_total_value(
                self.prev_enemy_value_snapshot[p_id]
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
                self._episode_pylon_max_vitality[p_id] = pylon_max_vitality(own_pylon)
            if enemy_pylon is not None:
                enemy_hp = float(enemy_pylon.health) + float(
                    getattr(enemy_pylon, "shield", 0.0) or 0.0
                )
                self._episode_pylon_max_vitality[enemy_owner] = pylon_max_vitality(
                    enemy_pylon
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

    def squad_action_masks(
        self,
        unit_mask: np.ndarray,
        obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Flat mask for the squad ``[actors | command | target | spatial]`` vector.

        Only the **actor** and **command** slices carry information here.
        Actor/target legality that depends on the sampled command lives in the
        observation (``heal_actor_mask`` / ``heal_target_mask`` / ``enemy_mask``
        / ``command_modes`` / AoE / unload / cloak masks) and is applied inside
        ``SquadRecurrentPolicy``, because the command has not been sampled yet
        when this mask is built.

        Command legality gates a head whose prerequisites are entirely absent:
        ``heal_unit`` needs a ready healer *and* a wounded compatible ally,
        ``load_unit`` needs a load-capable actor *and* a compatible partner,
        ``attack_unit`` / ``attack_local`` need a weapon-ready attacker,
        ``attack_unit`` also needs a hostile inside that attacker's tactical
        crop, ``attack_local`` /
        ``aoe_local`` a mobile *or building* within tactical half-width of any
        living ally (``engage_enemy_mask`` — buildings are not pointer slots but
        spatial attack already hard-targets them), ``aoe_local`` also a ready
        caster (Viper needs air in that engage fovea), ``snipe`` a ready snipe
        caster *and* an eligible hostile inside that caster's tactical crop
        (bio for Ghost, energy unit for Feedback, any mobile for Yamato /
        Neural),
        ``unload_local`` a loaded transport, ``cloak_burrow`` a unit that can
        conceal, and ``uncloak_unburrow`` a currently concealed unit.
        """
        from src.agent.squad_constants import (
            CMD_MODE_ALLY_TARGET,
            CMD_MODE_AOE,
            CMD_MODE_ATTACK_LOCAL,
            CMD_MODE_CLOAK,
            CMD_MODE_ENEMY_TARGET,
            CMD_MODE_LOAD_TARGET,
            CMD_MODE_SNIPE,
            CMD_MODE_UNCLOAK,
            CMD_MODE_UNLOAD,
            MAX_SQUAD_UNITS,
            SQUAD_FLAT_ACTION_DIM,
            SQUAD_MASK_COMMAND_OFFSET,
            command_target_modes,
        )
        from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

        mask = np.ones(SQUAD_FLAT_ACTION_DIM, dtype=bool)
        unit_mask = np.asarray(unit_mask, dtype=np.float32).reshape(-1)
        for i in range(min(MAX_SQUAD_UNITS, len(unit_mask))):
            if unit_mask[i] < 0.5:
                # Dead/padded slot: force "not selected". Masking BOTH options False
                # makes MaskableCategorical fall back to uniform 50/50 sampling.
                mask[i * 2] = True
                mask[i * 2 + 1] = False

        action_heads = list(self.training_conf.get("action_heads") or [])
        categorical_heads = list(self.training_conf.get("categorical_heads") or [])
        n_configured = min(
            FIXED_UNIVERSAL_COMMAND_DIM, len(action_heads) + len(categorical_heads)
        )
        cmd_start = SQUAD_MASK_COMMAND_OFFSET
        mask[cmd_start : cmd_start + FIXED_UNIVERSAL_COMMAND_DIM] = False
        mask[cmd_start : cmd_start + n_configured] = True

        if obs is not None:
            modes = obs.get("command_modes")
            if modes is None:
                modes = command_target_modes(action_heads, categorical_heads)
            else:
                modes = np.asarray(modes)
            has_heal = bool(
                np.any(np.asarray(obs.get("heal_actor_mask", 0.0)) > 0.5)
                and np.any(np.asarray(obs.get("heal_target_mask", 0.0)) > 0.5)
            )
            has_load = bool(
                np.any(np.asarray(obs.get("load_actor_mask", 0.0)) > 0.5)
                and np.any(np.asarray(obs.get("load_target_mask", 0.0)) > 0.5)
            )
            has_attack_ready = bool(
                np.any(np.asarray(obs.get("attack_actor_mask", 0.0)) > 0.5)
            )
            has_enemy = bool(np.any(np.asarray(obs.get("enemy_mask", 0.0)) > 0.5))
            # Prefer engage (any-ally fovea); fall back to hard-anchor crop for
            # older obs dicts / tests that only populate tactical_enemy_mask.
            engage = obs.get("engage_enemy_mask", None)
            if engage is None:
                engage = obs.get("tactical_enemy_mask", 0.0)
            has_engage_enemy = bool(np.any(np.asarray(engage) > 0.5))
            has_aoe = bool(np.any(np.asarray(obs.get("aoe_actor_mask", 0.0)) > 0.5))
            has_snipe = bool(
                np.any(np.asarray(obs.get("snipe_actor_mask", 0.0)) > 0.5)
                and np.any(np.asarray(obs.get("snipe_target_mask", 0.0)) > 0.5)
            )
            has_unload = bool(
                np.any(np.asarray(obs.get("unload_actor_mask", 0.0)) > 0.5)
            )
            has_cloak = bool(
                np.any(np.asarray(obs.get("cloak_actor_mask", 0.0)) > 0.5)
            )
            has_uncloak = bool(
                np.any(np.asarray(obs.get("uncloak_actor_mask", 0.0)) > 0.5)
            )
            for idx in range(n_configured):
                mode = int(round(float(modes[idx])))
                if mode == CMD_MODE_ALLY_TARGET and not has_heal:
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_LOAD_TARGET and not has_load:
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_ENEMY_TARGET and (
                    not has_enemy or not has_attack_ready
                ):
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_SNIPE and not has_snipe:
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_ATTACK_LOCAL and (
                    not has_engage_enemy or not has_attack_ready
                ):
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_AOE and (
                    not has_aoe or not has_engage_enemy
                ):
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_UNLOAD and not has_unload:
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_CLOAK and not has_cloak:
                    mask[cmd_start + idx] = False
                elif mode == CMD_MODE_UNCLOAK and not has_uncloak:
                    mask[cmd_start + idx] = False
            if not mask[cmd_start : cmd_start + n_configured].any():
                # Never hand the policy a fully-masked command slice.
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
        target_unit_tag: Optional[int] = None,
    ) -> List[int]:
        """Apply one squad group order to selected units.

        Spatial commands share one decoded world ``Point2`` (agent-faithful: the
        clicked cell is the order target). Unit-pointer commands (``heal_unit``,
        ``load_unit``, ``attack_unit``, ``snipe``) instead take ``target_unit_tag``,
        the SC2 tag the policy selected through the terminal target slot.

        ``perspective_player_id`` is the env's SC2 owner for this order (trainee
        or opponent). Spatial cells are world-frame for every owner: no 180°
        un-rotation. Targeting uses each caster's ``owner_id``.

        Returns tags that must be skipped by last-order reissue this physics step:
        units that received a fresh order, plus heal/snipe/aoe lock holds. Selected
        units that no-op (null target, missing unit, illegal caster) are *not*
        returned — otherwise ``opponent_decision_interval`` hold steps leave them
        idle forever with no ``_unit_last_order`` to reissue.
        """
        if not self.agent_system or "active_agent" not in self.agent_system.unit_agents:
            return []

        action_heads = list(self.training_conf.get("action_heads") or [])
        categorical_heads = list(self.training_conf.get("categorical_heads") or [])
        n_spatial = len(action_heads)
        n_total = n_spatial + len(categorical_heads)
        if command_idx < 0 or command_idx >= n_total:
            return []

        is_categorical = command_idx >= n_spatial
        if is_categorical:
            head_name = categorical_heads[command_idx - n_spatial]
        else:
            head_name = action_heads[command_idx]

        anchor = self.all_units.find_by_tag(anchor_tag)
        if anchor is None and selected_tags:
            anchor = self.all_units.find_by_tag(selected_tags[0])
        if anchor is None:
            return []

        # CoM / role shaping in calculate_squad_step_reward keys off active_unit_tag.
        self.active_unit_tag = int(anchor.tag)

        agent = self.agent_system.unit_agents["active_agent"]
        game_info = getattr(self, "game_info", None)
        if game_info is not None:
            agent.last_game_info = game_info

        ordered: List[int] = []

        # Categorical cloak/burrow: no spatial decode.
        if is_categorical and head_name in ("cloak_burrow", "uncloak_unburrow"):
            conceal = head_name == "cloak_burrow"
            metric_kind = "cloak" if conceal else "uncloak"
            for unit_tag in selected_tags:
                unit = self.all_units.find_by_tag(unit_tag)
                if unit is None or unit.type_id == UnitTypeId.SHIELDBATTERY:
                    continue
                if not neural_command_allowed(unit, perspective_player_id, head_name):
                    continue
                # Cloak/burrow cancels Yamato prep / Steady Targeting / AoE cast.
                if self._refresh_snipe_lock(unit) or self._refresh_aoe_lock(unit):
                    ordered.append(int(unit_tag))
                    continue
                self._mark_command_attempt(unit, metric_kind)
                self._issue_cloak_burrow(unit, conceal=conceal)
                self._unit_last_order[int(unit_tag)] = (head_name, None)
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
                ordered.append(int(unit_tag))
            return ordered

        if is_categorical:
            return []  # Unknown categorical head — no-op

        if head_name in UNIT_TARGET_COMMAND_HEADS:
            return self._apply_unit_target_command(
                selected_tags,
                head_name=head_name,
                target_unit_tag=target_unit_tag,
                perspective_player_id=perspective_player_id,
            )

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
            if not neural_command_allowed(unit, perspective_player_id, head_name):
                continue
            if "heal" in head_name:
                self._mark_heal_command_attempt(unit)
                # Misses are no-ops, not moves: cooldown-retreat must not treat
                # a failed Transfuse as a kite. Do not claim ordered either —
                # that would block last-order reissue of a prior move/attack.
                if self._issue_heal_at_point(unit, target) is None:
                    continue
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
                ordered.append(int(unit_tag))
            elif self._refresh_heal_lock(unit):
                # Walk-in + Transfuse/Heal start still inside the lock window.
                ordered.append(int(unit_tag))
            elif self._refresh_snipe_lock(unit):
                # Yamato / Steady Targeting / Neural prep must finish.
                ordered.append(int(unit_tag))
            elif "aoe" in head_name:
                self._mark_command_attempt(unit, "aoe")
                if not self._issue_aoe_with_lock(unit, target):
                    continue
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
                ordered.append(int(unit_tag))
            elif self._refresh_aoe_lock(unit):
                # Corrosive Bile / Psi Storm / Fungal cast must finish.
                ordered.append(int(unit_tag))
            elif "unload" in head_name:
                self._mark_command_attempt(unit, "unload")
                if not self._issue_unload_at_point(unit, target):
                    continue
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
                ordered.append(int(unit_tag))
            elif "attack" in head_name:
                issue_attack_at_point(self, unit, target)
                self._mark_command_attempt(unit, "attack")
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=True)
                ordered.append(int(unit_tag))
            else:
                self.do(unit.move(target))
                self._mark_command_attempt(unit, "move")
                self._unit_last_order[int(unit_tag)] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=True, attacked=False)
                ordered.append(int(unit_tag))
        return ordered

    def _apply_unit_target_command(
        self,
        selected_tags: List[int],
        *,
        head_name: str,
        target_unit_tag: Optional[int],
        perspective_player_id: Optional[int] = None,
    ) -> List[int]:
        """Issue ``heal_unit`` / ``load_unit`` / ``attack_unit`` / ``snipe`` against one tag.

        The policy already chose *which* unit through the terminal target slot, so
        there is no cell geometry to get wrong and no hidden retargeting: the
        order goes to exactly that tag. Per-caster legality is still re-checked
        (an SCV cannot repair a Marine; a full Medivac cannot load; Ghost Snipe
        needs bio; a new snipe or ``attack_unit`` target must lie in that
        actor's tactical crop), because the observation's target masks only
        guarantee *some* actor in the squad matches.

        Returns tags that received an order or are held under an in-flight
        heal/snipe/aoe lock. A null/missing target, or a per-caster no-op
        (full-HP heal, friendly-fire attack, failed snipe), is not ordered so
        reissue can keep prior move/attack orders alive.
        """
        if target_unit_tag is None:
            return []
        target = self.all_units.find_by_tag(int(target_unit_tag))
        if target is None:
            return []
        is_heal = head_name in ALLY_TARGET_COMMAND_HEADS
        is_load = head_name in LOAD_TARGET_COMMAND_HEADS
        is_snipe = head_name in SNIPE_COMMAND_HEADS
        ordered: List[int] = []
        for unit_tag in selected_tags:
            unit = self.all_units.find_by_tag(unit_tag)
            if unit is None or unit.type_id == UnitTypeId.SHIELDBATTERY:
                continue
            if not neural_command_allowed(unit, perspective_player_id, head_name):
                continue
            if int(getattr(unit, "tag", 0) or 0) == int(target_unit_tag):
                continue  # never order a unit onto itself
            if is_snipe:
                if self._refresh_aoe_lock(unit):
                    # Snipe would cancel bile / storm / fungal wind-up.
                    ordered.append(int(unit_tag))
                    continue
                self._mark_command_attempt(unit, "snipe")
                if not self._issue_snipe_with_lock(unit, target):
                    continue
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
            elif self._refresh_snipe_lock(unit):
                # attack_unit / heal_unit / load would cancel Yamato prep.
                ordered.append(int(unit_tag))
                continue
            elif self._refresh_aoe_lock(unit):
                # attack_unit / heal_unit / load would cancel AoE cast.
                ordered.append(int(unit_tag))
                continue
            elif is_heal:
                self._mark_heal_command_attempt(unit)
                if self._issue_heal_on_unit(unit, target) is None:
                    continue
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
            elif is_load:
                self._mark_command_attempt(unit, "load")
                if not self._issue_load_on_unit(unit, target):
                    continue
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
            else:
                self._mark_command_attempt(unit, "attack")
                if not self._issue_attack_on_unit(unit, target):
                    continue
                self._set_last_order_flags(unit_tag, moved=False, attacked=True)
            self._unit_last_order[int(unit_tag)] = (head_name, target.position)
            ordered.append(int(unit_tag))
        return ordered

    def _issue_attack_on_unit(self, unit: Unit, target: Unit) -> bool:
        """Focus-fire ``target``; skip friendly fire under ``debug_control_enemy``.

        Both armies look allied to the P1 client after ``debug_control_enemy``, so
        hostility must be decided by ``owner_id`` — an alliance-bucket check would
        happily let the opponent shoot its own army and inflate damage rewards.
        A body under Neural Parasite stays a legal target for both owners.

        Weaponless casters whose spatial attack already remaps (Swarm Host →
        Spawn Locusts, Disruptor → Purification Nova) cast at ``target.position``
        so ``attack_unit`` stays agent-faithful to the same ability mapping.
        A Corruptor aimed at a building casts Caustic Spray, matching the
        spatial attack on a structure footprint.

        Returns ``True`` when an order was issued. A target outside this
        shooter's tactical crop is refused so the unit is not walked across
        the map.
        """
        if int(getattr(target, "owner_id", 0) or 0) == int(
            getattr(unit, "owner_id", 0) or 0
        ) and not unit_is_neuraled(target):
            return False
        if not unit_in_tactical_fovea(unit, target):
            return False
        if uses_spawn_locusts_for_attack(unit):
            self.do(unit(AbilityId.EFFECT_SPAWNLOCUSTS, target.position))
            return True
        if uses_purification_nova_for_attack(unit):
            self.do(unit(AbilityId.EFFECT_PURIFICATIONNOVA, target.position))
            return True
        if uses_caustic_spray_on_structures(unit) and bool(
            getattr(target, "is_structure", False)
        ):
            self.do(unit(AbilityId.CAUSTICSPRAY_CAUSTICSPRAY, target))
            return True
        self.do(unit.attack(target))
        return True

    def _issue_heal_on_unit(self, healer: Unit, patient: Unit) -> Optional[str]:
        """Heal/Repair/Transfuse ``patient`` directly, bookkeeping casts as usual."""
        if int(getattr(patient, "owner_id", 0) or 0) != int(
            getattr(healer, "owner_id", 0) or 0
        ):
            return None
        if getattr(patient, "is_structure", False):
            return None
        if not heal_target_is_eligible(healer, patient):
            return None
        if _missing_health(patient) <= 0.0:
            return None
        ability = self._HEAL_ABILITY_BY_TYPE.get(healer.type_id)
        if ability is None:
            return None
        self.do(healer(ability, patient))
        self._record_heal_cast(healer, patient)
        return "cast"

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
        """Biology compatibility of a heal order (see ``heal_target_is_eligible``)."""
        return heal_target_is_eligible(healer, ally)

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

    @staticmethod
    def _snipe_caster_move_speed(unit: Unit) -> float:
        for attr in ("real_speed", "movement_speed"):
            try:
                speed = float(getattr(unit, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                speed = 0.0
            if speed > 1e-6:
                return speed
        defaults = CaptureTheFlagBot._DEFAULT_SNIPE_CASTER_MOVE_SPEED
        return float(defaults.get(getattr(unit, "type_id", None), 2.5))

    def _snipe_lock_until_loop(self, caster: Unit, target: Unit) -> int:
        """Game loop when walk-in + Yamato/Snipe prep animation should finish."""
        now = CaptureTheFlagBot._game_loop(self)
        type_id = getattr(caster, "type_id", None)
        anim = float(
            SNIPE_CAST_ANIMATION_SECONDS.get(
                type_id, CaptureTheFlagBot._DEFAULT_SNIPE_CAST_ANIMATION_SECONDS
            )
        )
        try:
            cast_range = float(SNIPE_CAST_RANGE.get(type_id, 0.0) or 0.0)
        except (TypeError, ValueError):
            cast_range = 0.0
        extra = 0.0
        try:
            dist = float(caster.position.distance_to(target.position))
            extra = max(0.0, dist - cast_range)
        except (TypeError, ValueError, AttributeError):
            extra = 0.0
        speed = CaptureTheFlagBot._snipe_caster_move_speed(caster)
        walk = extra / max(speed, 1e-6) if speed > 1e-6 else extra
        loops = int(
            math.ceil((walk + anim) * CaptureTheFlagBot._HEAL_GAME_LOOPS_PER_SECOND)
        )
        return now + max(loops, 1)

    def _refresh_snipe_lock(self, unit: Unit) -> bool:
        """True while walk-in + snipe prep must not be overwritten.

        Yamato's ~3s laser and Ghost Steady Targeting are cancelled by any new
        order (and SC2 still burns the full Yamato cooldown on prep cancel).
        Lock until the window elapses, the caster goes idle, or the target dies.
        ``snipe`` is still allowed (reaffirm / retarget).
        """
        tag = int(getattr(unit, "tag", 0) or 0)
        lock = getattr(self, "_snipe_lock", None)
        if tag <= 0 or not isinstance(lock, dict) or tag not in lock:
            return False
        entry = lock[tag]
        try:
            target_tag = int(entry[0])
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
        target = self.all_units.find_by_tag(target_tag)
        if target is None:
            lock.pop(tag, None)
            return False
        return True

    def _continuing_snipe_lock(self, unit: Unit, target: Unit) -> bool:
        """True when this snipe reaffirms the same in-flight target."""
        tag = int(getattr(unit, "tag", 0) or 0)
        target_tag = int(getattr(target, "tag", 0) or 0)
        lock = getattr(self, "_snipe_lock", None)
        if tag <= 0 or target_tag <= 0 or not isinstance(lock, dict):
            return False
        entry = lock.get(tag)
        if entry is None:
            return False
        try:
            return (
                int(entry[0]) == target_tag
                and CaptureTheFlagBot._game_loop(self) < int(entry[1])
            )
        except (TypeError, ValueError, IndexError):
            return False

    def _record_snipe_cast(self, unit: Unit, target: Unit) -> None:
        """Bookkeep walk-in + prep lock after a successful ``snipe`` issue."""
        tag = int(getattr(unit, "tag", 0) or 0)
        target_tag = int(getattr(target, "tag", 0) or 0)
        if not tag or not target_tag:
            return
        lock = getattr(self, "_snipe_lock", None)
        if not isinstance(lock, dict):
            self._snipe_lock = {}
            lock = self._snipe_lock
        lock[tag] = (
            target_tag,
            CaptureTheFlagBot._snipe_lock_until_loop(self, unit, target),
        )

    def _issue_snipe_with_lock(self, unit: Unit, target: Unit) -> bool:
        """Issue ``snipe`` and lock the caster through walk-in + prep.

        Reaffirming the same target while locked does not restamp the long
        Yamato cooldown (would strand the mask closed if prep was cancelled).
        """
        continuing = CaptureTheFlagBot._continuing_snipe_lock(self, unit, target)
        issued = issue_snipe_on_unit(
            self,
            unit,
            target,
            stamp_cooldown=not continuing,
            require_fovea=not continuing,
        )
        if issued:
            CaptureTheFlagBot._record_snipe_cast(self, unit, target)
            if not continuing:
                self._credit_snipe_target_value(target)
        return issued

    @staticmethod
    def _aoe_caster_move_speed(unit: Unit) -> float:
        for attr in ("real_speed", "movement_speed"):
            try:
                speed = float(getattr(unit, attr, 0.0) or 0.0)
            except (TypeError, ValueError):
                speed = 0.0
            if speed > 1e-6:
                return speed
        defaults = CaptureTheFlagBot._DEFAULT_AOE_CASTER_MOVE_SPEED
        return float(defaults.get(getattr(unit, "type_id", None), 2.5))

    def _aoe_lock_until_loop(self, caster: Unit, target_pos: Point2) -> int:
        """Game loop when walk-in + AoE cast animation should finish."""
        now = CaptureTheFlagBot._game_loop(self)
        type_id = getattr(caster, "type_id", None)
        anim = float(
            AOE_CAST_ANIMATION_SECONDS.get(
                type_id, CaptureTheFlagBot._DEFAULT_AOE_CAST_ANIMATION_SECONDS
            )
        )
        try:
            cast_range = float(AOE_CAST_RANGE.get(type_id, 0.0) or 0.0)
        except (TypeError, ValueError):
            cast_range = 0.0
        extra = 0.0
        try:
            dist = float(caster.position.distance_to(target_pos))
            extra = max(0.0, dist - cast_range)
        except (TypeError, ValueError, AttributeError):
            extra = 0.0
        speed = CaptureTheFlagBot._aoe_caster_move_speed(caster)
        walk = extra / max(speed, 1e-6) if speed > 1e-6 else extra
        loops = int(
            math.ceil((walk + anim) * CaptureTheFlagBot._HEAL_GAME_LOOPS_PER_SECOND)
        )
        return now + max(loops, 1)

    def _refresh_aoe_lock(self, unit: Unit) -> bool:
        """True while walk-in + AoE cast must not be overwritten.

        Corrosive Bile / Psi Storm / Fungal / EMP / Parasitic Bomb are cancelled
        by a new order. Lock until the window elapses, the caster goes idle, or
        (Viper) the hard-target dies. ``aoe_local`` is still allowed (reaffirm /
        retarget).
        """
        tag = int(getattr(unit, "tag", 0) or 0)
        lock = getattr(self, "_aoe_lock", None)
        if tag <= 0 or not isinstance(lock, dict) or tag not in lock:
            return False
        entry = lock[tag]
        try:
            lock_key = entry[0]
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
        if isinstance(lock_key, int):
            target = self.all_units.find_by_tag(int(lock_key))
            if target is None:
                lock.pop(tag, None)
                return False
        return True

    def _continuing_aoe_lock(self, unit: Unit, target: Point2) -> bool:
        """True when this ``aoe_local`` reaffirms the same in-flight cast."""
        tag = int(getattr(unit, "tag", 0) or 0)
        lock = getattr(self, "_aoe_lock", None)
        if tag <= 0 or not isinstance(lock, dict):
            return False
        entry = lock.get(tag)
        if entry is None:
            return False
        try:
            lock_key = entry[0]
            until_loop = int(entry[1])
        except (TypeError, ValueError, IndexError):
            return False
        if CaptureTheFlagBot._game_loop(self) >= until_loop:
            return False
        if isinstance(lock_key, int):
            hard = resolve_air_hostile_at_point(self, unit, target)
            return hard is not None and int(getattr(hard, "tag", 0) or 0) == int(
                lock_key
            )
        try:
            return (
                float(lock_key.x) == float(target.x)
                and float(lock_key.y) == float(target.y)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    def _record_aoe_cast(self, unit: Unit, target: Point2) -> None:
        """Bookkeep walk-in + cast lock after a successful ``aoe_local`` issue."""
        tag = int(getattr(unit, "tag", 0) or 0)
        if not tag:
            return
        lock = getattr(self, "_aoe_lock", None)
        if not isinstance(lock, dict):
            self._aoe_lock = {}
            lock = self._aoe_lock
        lock_key: Any = Point2((float(target.x), float(target.y)))
        ref_pos = target
        if getattr(unit, "type_id", None) == UnitTypeId.VIPER:
            hard = resolve_air_hostile_at_point(self, unit, target)
            if hard is not None:
                lock_key = int(getattr(hard, "tag", 0) or 0)
                ref_pos = hard.position
        lock[tag] = (
            lock_key,
            CaptureTheFlagBot._aoe_lock_until_loop(self, unit, ref_pos),
        )

    def _issue_aoe_with_lock(self, unit: Unit, target: Point2) -> bool:
        """Issue ``aoe_local`` and lock the caster through walk-in + cast.

        Reaffirming the same point / air target while locked does not restamp
        the local ability cooldown.
        """
        continuing = CaptureTheFlagBot._continuing_aoe_lock(self, unit, target)
        issued = issue_aoe_at_point(
            self, unit, target, stamp_cooldown=not continuing
        )
        if issued:
            CaptureTheFlagBot._record_aoe_cast(self, unit, target)
            if not continuing:
                CaptureTheFlagBot._credit_aoe_splash_value(self, unit, target)
        return issued

    def _credit_aoe_splash_value(self, unit: Unit, target: Point2) -> None:
        """Credit economic value of hostiles in this caster's splash radius.

        Bile is scored at cast with its impact radius (travel delay ignored).
        Viper Parasitic Bomb uses the resolved air unit as the splash center.
        """
        radius = float(AOE_SPLASH_RADIUS.get(getattr(unit, "type_id", None), 0.0) or 0.0)
        if radius <= 0.0:
            return
        center = target
        if getattr(unit, "type_id", None) == UnitTypeId.VIPER:
            hard = resolve_air_hostile_at_point(self, unit, target)
            if hard is None:
                return
            center = getattr(hard, "position", None) or target
        cx = float(getattr(center, "x", 0.0))
        cy = float(getattr(center, "y", 0.0))
        r_conf = self.training_conf.get("rewards", {})
        econ = EconomicRewardConfig.from_rewards_conf(r_conf)
        total = 0.0
        for hostile in _hostile_candidates(self, unit):
            if hostile is None:
                continue
            pos = getattr(hostile, "position", None)
            if pos is None:
                continue
            dist = math.hypot(float(pos.x) - cx, float(pos.y) - cy)
            if dist > radius + 1e-6:
                continue
            total += unit_economic_value(
                hostile,
                vespene_weight=econ.vespene_weight,
                value_norm=econ.value_norm,
            )
        try:
            prev = float(getattr(self, "_step_aoe_land_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            prev = 0.0
        self._step_aoe_land_value = prev + total

    _HEAL_ABILITY_BY_TYPE: Dict[UnitTypeId, AbilityId] = {
        UnitTypeId.MEDIVAC: AbilityId.MEDIVACHEAL_HEAL,
        UnitTypeId.SCV: AbilityId.EFFECT_REPAIR_SCV,
        UnitTypeId.QUEEN: AbilityId.TRANSFUSION_TRANSFUSION,
    }

    def _record_heal_cast(self, unit: Unit, heal_target: Unit) -> None:
        """Bookkeep a just-issued heal order (cast metrics, walk-in lock, cooldown).

        A reissue on the *same* patient while the previous cast's walk-in /
        animation lock is still active is the engine reaffirming one in-flight
        heal, not a new cast. Crediting ``heal_cast_reward`` on every squad step
        the policy keeps clicking pays *per step parked on heal*, not per cast
        landed, which let healing out-earn combat by camping a wounded ally.
        Only a fresh engagement (new patient, or the previous lock lapsed) is
        cast-bonus eligible; ``heal_reward``'s per-HP shaping uses the broader
        ``_heal_order_tags`` so a Medivac beam / SCV repair keeps earning HP
        credit on every step it actually ticks HP.
        """
        tag = int(getattr(unit, "tag", 0) or 0)
        if not tag:
            return
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
        ability = self._HEAL_ABILITY_BY_TYPE.get(unit.type_id)
        if ability is None:
            return None

        heal_target = self._resolve_heal_target_in_cell(unit, target)
        if heal_target is None:
            return None
        self.do(unit(ability, heal_target))
        self._record_heal_cast(unit, heal_target)
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

    def _issue_load_on_unit(self, unit: Unit, target: Unit) -> bool:
        """Dual-role load by tag: transport LOADs passenger; passenger smarts onto transport.

        Returns ``True`` when an order was issued.
        """
        from src.utils.unit_feature_helpers import load_pair_is_eligible, unit_is_transport

        if not load_pair_is_eligible(unit, target):
            return False
        if unit_is_transport(unit):
            ability = self._TRANSPORT_LOAD_ABILITY.get(unit.type_id)
            if ability is None:
                return False
            self.do(unit(ability, target))
            return True
        self.do(unit.smart(target))
        return True

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
        """Legacy cell-click load (tests / non-squad paths). Prefer ``_issue_load_on_unit``."""
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

    def _issue_unload_at_point(self, unit: Unit, target: Point2) -> bool:
        """Unload all cargo at the decoded world point (transports with cargo only).

        Returns ``True`` when an unload order was issued.
        """
        if int(getattr(unit, "cargo_max", 0) or 0) <= 0:
            return False
        if int(getattr(unit, "cargo_used", 0) or 0) <= 0:
            return False
        ability = self._TRANSPORT_UNLOAD_ABILITY.get(unit.type_id, AbilityId.UNLOADALLAT)
        if ability == AbilityId.UNLOADALL_WARPPRISM:
            self.do(unit(ability))
        else:
            self.do(unit(ability, target))
        return True

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

    def _issue_cloak_burrow(self, unit: Unit, *, conceal: bool) -> bool:
        """Shared categorical head: cloak↔burrow / uncloak↔unburrow by unit type.

        Returns True when an order was actually issued. Conceal shaping is
        credited here, before physics, so the bonus sees the pre-step cloak,
        burrow, and health state.
        """
        tid = unit.type_id
        if tid == UnitTypeId.DARKTEMPLAR:
            return False  # permanently cloaked
        if tid == UnitTypeId.GHOST:
            ability = (
                AbilityId.BEHAVIOR_CLOAKON_GHOST
                if conceal
                else AbilityId.BEHAVIOR_CLOAKOFF_GHOST
            )
            self.do(unit(ability))
            self._credit_conceal_command(unit, conceal=conceal)
            return True
        if tid == UnitTypeId.BANSHEE:
            ability = (
                AbilityId.BEHAVIOR_CLOAKON_BANSHEE
                if conceal
                else AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE
            )
            self.do(unit(ability))
            self._credit_conceal_command(unit, conceal=conceal)
            return True
        if tid == UnitTypeId.MOTHERSHIP:
            if conceal:
                self.do(unit(AbilityId.MOTHERSHIPCLOAK_ORACLECLOAKFIELD))
                self._credit_conceal_command(unit, conceal=True)
                return True
            # No reliable cancel id for mothership field in LotV API — no-op on reveal.
            return False
        if conceal:
            ability = self._BURROW_DOWN_BY_TYPE.get(tid)
        else:
            ability = self._BURROW_UP_BY_TYPE.get(tid)
        if ability is None:
            return False
        self.do(unit(ability))
        self._credit_conceal_command(unit, conceal=conceal)
        return True

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
                    if self._refresh_snipe_lock(unit) or self._refresh_aoe_lock(unit):
                        return
                    conceal = head_name == "cloak_burrow"
                    self._mark_command_attempt(
                        unit, "cloak" if conceal else "uncloak"
                    )
                    self._issue_cloak_burrow(unit, conceal=conceal)
                    self._unit_last_order[unit_tag] = (head_name, None)
                    self._set_last_order_flags(unit_tag, moved=False, attacked=False)
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
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
            elif self._refresh_heal_lock(unit):
                return
            elif self._refresh_snipe_lock(unit):
                return
            elif "aoe" in head_name:
                self._mark_command_attempt(unit, "aoe")
                self._issue_aoe_with_lock(unit, target)
                self._unit_last_order[unit_tag] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
            elif self._refresh_aoe_lock(unit):
                return
            elif "unload" in head_name:
                self._mark_command_attempt(unit, "unload")
                self._issue_unload_at_point(unit, target)
                self._unit_last_order[unit_tag] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=False)
            elif "attack" in head_name:
                issue_attack_at_point(self, unit, target)
                self._mark_command_attempt(unit, "attack")
                self._unit_last_order[unit_tag] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=False, attacked=True)
            else:
                self.do(unit.move(target))
                self._mark_command_attempt(unit, "move")
                self._unit_last_order[unit_tag] = (head_name, target)
                self._set_last_order_flags(unit_tag, moved=True, attacked=False)
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
            self._set_last_order_flags(
                unit_tag, moved=not is_attack, attacked=is_attack
            )

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

    def _set_last_order_flags(
        self, unit_tag: int, *, moved: bool = False, attacked: bool = False
    ) -> None:
        """Record whether this unit's latest policy/reissue order was move or attack."""
        tag = int(unit_tag)
        self.last_action_was_move_by_unit[tag] = bool(moved)
        self.last_action_was_attack_by_unit[tag] = bool(attacked)

    def record_pre_action_retreat_state(self, unit_tag: int) -> None:
        """
        Snapshot pre-action state for per-step distance shaping.

        Captures weapon cooldown / enemy-CoM (cooldown retreat) and distance to
        the defender pylon (defender zone penalty). The enemy CoM point is frozen
        so post-step chase by faster melee cannot erase retreat credit.

        Also snapshots whether the unit's last order was an attack — paired with
        cooldown, that opens the kite window even when ``weapon_cooldown`` has
        already returned to 0 across a long physics step.
        """
        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
            self._pre_action_enemy_com.pop(unit_tag, None)
            self._pre_action_attacked_prev_step.pop(unit_tag, None)
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
            self._pre_action_vitality.pop(unit_tag, None)
            self._pre_action_nearest_enemy.pop(unit_tag, None)
            self._pre_action_dist_to_nearest_enemy.pop(unit_tag, None)
            self._pre_action_spell_energy.pop(unit_tag, None)
            return

        # python-sc2 returns -1 when the unit cannot attack.
        try:
            raw_cd = float(getattr(unit, "weapon_cooldown", 0.0))
        except (TypeError, ValueError):
            raw_cd = 0.0
        self._pre_action_weapon_cd_by_unit[unit_tag] = max(0.0, raw_cd)
        self._pre_action_attacked_prev_step[unit_tag] = bool(
            self.last_action_was_attack_by_unit.get(unit_tag, False)
        )
        self._pre_action_vitality[unit_tag] = self._raw_vitality(unit)

        enemies = self._enemies_for_unit(unit)
        enemy_com = self._enemy_com_in_tactical_fovea(unit, enemies)
        if enemy_com is None:
            self._pre_action_dist_to_enemy_com.pop(unit_tag, None)
            self._pre_action_enemy_com.pop(unit_tag, None)
        else:
            self._pre_action_enemy_com[unit_tag] = enemy_com
            self._pre_action_dist_to_enemy_com[unit_tag] = self._dist_to_point(
                unit, enemy_com
            )

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

        self._snapshot_spellcaster_spacing(unit)

    @staticmethod
    def _raw_vitality(unit: Any) -> float:
        """Current health + shields (0 when the unit or bars are missing)."""
        if unit is None:
            return 0.0
        try:
            health = float(getattr(unit, "health", 0.0) or 0.0)
            shield = float(getattr(unit, "shield", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, health + shield)

    @staticmethod
    def _combined_vitality(unit: Any) -> float:
        """Fraction of max health+shields in ``[0, 1]``."""
        curr = CaptureTheFlagBot._raw_vitality(unit)
        if unit is None:
            return 0.0
        try:
            hmax = float(getattr(unit, "health_max", 0.0) or 0.0)
            smax = float(getattr(unit, "shield_max", 0.0) or 0.0)
        except (TypeError, ValueError):
            hmax, smax = 0.0, 0.0
        cap = hmax + smax
        if cap <= 1e-6:
            cap = max(curr, 1.0)
        return max(0.0, min(1.0, curr / cap))

    def calculate_cooldown_retreat_reward(
        self, unit_tag: int, enemies: Iterable[Unit]
    ) -> float:
        """
        Reward kiting away from the enemy CoM while under fire and low on HP.

        Gates (all required):
        - ``cooldown_retreat_scale`` > 0 in config
        - Unit has ranged weapon range (> ``_RANGED_WEAPON_RANGE_MIN``)
        - Vitality dropped since the pre-action snapshot (actually under attack)
        - This step's order was a move (not attack)
        - Enemies were in the unit's tactical fovea at pre-action (frozen CoM)
        - Distance to that **frozen** pre-action CoM increased

        Magnitude is ``scale * (1 - combined_vitality) * Δdist`` so a near-dead
        kite pays more than a full-HP one. Distance uses the frozen CoM so a
        faster melee chase in the same physics step cannot cancel the kite.
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
            self._pre_action_enemy_com.pop(unit_tag, None)
            self._pre_action_weapon_cd_by_unit.pop(unit_tag, None)
            self._pre_action_attacked_prev_step.pop(unit_tag, None)
            self._pre_action_vitality.pop(unit_tag, None)
            return 0.0

        if not self._is_ranged_combat_unit(unit, enemies):
            return 0.0

        prev_dist = self._pre_action_dist_to_enemy_com.get(unit_tag)
        pre_com = self._pre_action_enemy_com.get(unit_tag)
        prev_vit = self._pre_action_vitality.get(unit_tag)
        if prev_dist is None or pre_com is None or prev_vit is None:
            return 0.0

        eps = 1e-3
        curr_vit = self._raw_vitality(unit)
        if curr_vit >= float(prev_vit) - eps:
            return 0.0

        if not self.last_action_was_move_by_unit.get(unit_tag, False):
            return 0.0

        curr_dist = self._dist_to_point(unit, pre_com)
        if curr_dist <= prev_dist + eps:
            return 0.0

        hp_scale = 1.0 - self._combined_vitality(unit)
        if hp_scale <= 0.0:
            return 0.0
        return scale * hp_scale * (curr_dist - prev_dist)

    def calculate_defender_zone_penalty(
        self, unit_tag: int, own_pylon_pos: Optional[Point2]
    ) -> float:
        """
        Shape defenders that leave the pylon zone.

        Outside ``defender_zone_radius``: pay ``-defender_zone_leave_penalty``
        every step, plus ``defender_zone_return_scale * Δdist_in`` when moving
        closer to the pylon. No extra farther-from-pylon term (leave cost already
        charges standing or walking away). Inside the zone: 0.
        """
        if unit_tag is None:
            return 0.0

        if self.unit_command_map.get(unit_tag, "attack") != "defend":
            return 0.0

        r_conf = self.training_conf.get("rewards", {})
        leave = float(r_conf.get("defender_zone_leave_penalty", 0.08))
        return_scale = float(r_conf.get("defender_zone_return_scale", 1.0))
        zone_radius = float(r_conf.get("defender_zone_radius", 10.0))
        if leave == 0.0 and return_scale == 0.0:
            return 0.0

        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            self._pre_action_dist_to_defend_pylon.pop(unit_tag, None)
            return 0.0

        pylon_pos = self._defend_pylon_pos_for_unit(unit, own_pylon_pos)
        if pylon_pos is None:
            return 0.0

        curr_dist = self._dist_to_point(unit, pylon_pos)
        eps = 1e-3
        if curr_dist <= zone_radius + eps:
            return 0.0

        total = -leave
        prev_dist = self._pre_action_dist_to_defend_pylon.get(unit_tag)
        if prev_dist is not None and return_scale != 0.0 and curr_dist + eps < prev_dist:
            total += return_scale * (prev_dist - curr_dist)
        return total

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

    def _squad_objective_anchor(
        self,
        squad_tags: List[int],
        fallback: Optional[Point2],
    ) -> Optional[Point2]:
        for tag in squad_tags:
            mapped = self.unit_target_map.get(tag)
            if mapped is not None:
                return mapped
        return fallback

    def _enemy_value_position_scale(
        self,
        *,
        squad_is_defend: bool,
        squad_tags: List[int],
        own_pylon_pos: Optional[Point2],
        enemy_objective: Optional[Point2],
        r_conf: dict,
    ):
        """Scale enemy-value credit by distance to the role objective."""
        if squad_is_defend:
            inner = float(r_conf.get("defender_zone_radius", 10.0))
            anchor = self._defender_zone_anchor(squad_tags, own_pylon_pos)
        else:
            inner = float(r_conf.get("attacker_objective_radius", 12.0))
            anchor = self._squad_objective_anchor(squad_tags, enemy_objective)
        falloff = float(r_conf.get("objective_credit_falloff", 15.0))
        min_scale = float(r_conf.get("objective_credit_min_scale", 0.25))

        def _scale(pos):
            if pos is None or anchor is None:
                return 1.0
            dist = math.hypot(
                float(pos.x) - float(anchor.x),
                float(pos.y) - float(anchor.y),
            )
            return objective_credit_scale(
                dist, inner_radius=inner, falloff=falloff, min_scale=min_scale
            )

        return _scale

    @staticmethod
    def _unit_is_concealed(unit: Any) -> bool:
        return bool(getattr(unit, "is_cloaked", False)) or bool(
            getattr(unit, "is_burrowed", False)
        )

    def _allied_detector_covers(
        self, unit: Any, own_units: Iterable, own_structs: Iterable
    ) -> bool:
        pos = getattr(unit, "position", None)
        if pos is None:
            return False
        ux, uy = float(pos.x), float(pos.y)
        for detector in list(own_units or []) + list(own_structs or []):
            if detector is None:
                continue
            try:
                rng = float(getattr(detector, "detect_range", 0.0) or 0.0)
            except (TypeError, ValueError):
                rng = 0.0
            if rng <= 0.0:
                continue
            dpos = getattr(detector, "position", None)
            if dpos is None:
                continue
            if math.hypot(ux - float(dpos.x), uy - float(dpos.y)) <= rng + 1e-6:
                return True
        return False

    @staticmethod
    def _is_backline_spellcaster(unit: Any) -> bool:
        """Ghost, High Templar, Infestor (including burrowed), and Viper."""
        return getattr(unit, "type_id", None) in _BACKLINE_SPELLCASTER_TYPES

    @staticmethod
    def _is_swarm_host(unit: Any) -> bool:
        """Swarm Host, burrowed or not. Locusts are a separate summoned unit."""
        return getattr(unit, "type_id", None) in _SWARM_HOST_TYPES

    def _swarm_host_on_cooldown(self, unit: Any) -> bool:
        """True when this host's locust launch was cooling down before the step.

        Spawn Locusts reports through ``weapon_cooldown``. The pre-action
        snapshot wins, so a cooldown that expires during the physics step still
        counts as the state the host decided from. A missing snapshot falls
        back to the live cooldown.
        """
        tag = getattr(unit, "tag", None)
        try:
            tag = int(tag) if tag is not None else None
        except (TypeError, ValueError):
            tag = None
        stored = None
        if tag is not None:
            stored = self._pre_action_weapon_cd_by_unit.get(tag)
        raw = stored if stored is not None else getattr(unit, "weapon_cooldown", 0.0)
        try:
            return float(raw or 0.0) > 1e-6
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _spellcaster_energy_floor(type_id: Any) -> float:
        """Energy below which this caster has no offensive spell left.

        The floor is the cheaper of ``snipe`` and ``aoe_local`` (Ghost Snipe 50
        vs EMP 75, Feedback 50 vs Storm 75, Fungal 75 vs Neural 100, Viper
        Parasitic Bomb 125). At or above the floor the unit can still cast, so
        it is not paid to run.
        """
        costs = []
        for table in (_SNIPE_ENERGY_COST, _AOE_ENERGY_COST):
            raw = table.get(type_id)
            if raw is None:
                continue
            try:
                cost = float(raw)
            except (TypeError, ValueError):
                continue
            if cost > 0.0:
                costs.append(cost)
        if not costs:
            return 0.0
        return min(costs)

    @staticmethod
    def _safe_energy(unit: Any) -> Optional[float]:
        try:
            value = getattr(unit, "energy", None)
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _health_fraction(unit: Any) -> float:
        """Current health / max health in ``[0, 1]``. Shields are not health.

        Missing or non-numeric bars count as full, so a mock without a health
        bar does not look like a dying unit.
        """
        try:
            health = float(getattr(unit, "health", 0.0) or 0.0)
            health_max = float(getattr(unit, "health_max", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 1.0
        if health_max <= 1e-6:
            return 1.0
        return max(0.0, min(1.0, health / health_max))

    @staticmethod
    def _is_ambush_unit(unit: Any) -> bool:
        return getattr(unit, "type_id", None) in _AMBUSH_ATTACK_RANGE

    @staticmethod
    def _ambush_is_burrowed(unit: Any) -> bool:
        if getattr(unit, "type_id", None) in _AMBUSH_BURROWED_TYPES:
            return True
        return getattr(unit, "is_burrowed", False) is True

    def _ambush_attack_range(self, unit: Any) -> float:
        """Burrowed weapon range, or a larger range the unit already reports."""
        base = float(_AMBUSH_ATTACK_RANGE.get(getattr(unit, "type_id", None), 0.0))
        reported = 0.0
        for attr in ("ground_range", "air_range"):
            try:
                reported = max(reported, float(getattr(unit, attr, 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return max(base, reported)

    def _enemy_detector_covers(self, unit: Any) -> bool:
        """True when a hostile detector disc covers *unit*."""
        pos = getattr(unit, "position", None)
        if pos is None:
            return False
        ux, uy = float(pos.x), float(pos.y)
        for other in _hostile_candidates(self, unit):
            if other is None:
                continue
            try:
                rng = float(getattr(other, "detect_range", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if rng <= 0.0:
                continue
            opos = getattr(other, "position", None)
            if opos is None:
                continue
            if math.hypot(ux - float(opos.x), uy - float(opos.y)) <= rng + 1e-6:
                return True
        return False

    def _hostiles_in_ambush_range(self, unit: Any) -> int:
        """Non-structure hostiles inside this mine or lurker's attack range.

        Lurkers do not count air units. Widow Mines count both.
        """
        pos = getattr(unit, "position", None)
        if pos is None:
            return 0
        reach = self._ambush_attack_range(unit)
        if reach <= 0.0:
            return 0
        tid = getattr(unit, "type_id", None)
        ground_only = tid in _LURKER_AMBUSH_TYPES
        ux, uy = float(pos.x), float(pos.y)
        count = 0
        for other in _hostile_candidates(self, unit):
            if other is None or getattr(other, "is_structure", False) is True:
                continue
            if ground_only and getattr(other, "is_flying", False) is True:
                continue
            opos = getattr(other, "position", None)
            if opos is None:
                continue
            dist = math.hypot(ux - float(opos.x), uy - float(opos.y))
            if dist <= reach + 1e-6:
                count += 1
        return count

    def _nearest_hostile_mobile_point(self, unit: Any) -> Optional[Point2]:
        pos = getattr(unit, "position", None)
        if pos is None:
            return None
        ux, uy = float(pos.x), float(pos.y)
        best: Optional[Point2] = None
        best_d: Optional[float] = None
        for other in _hostile_candidates(self, unit):
            if other is None or getattr(other, "is_structure", False) is True:
                continue
            opos = getattr(other, "position", None)
            if opos is None:
                continue
            dist = math.hypot(ux - float(opos.x), uy - float(opos.y))
            if best_d is None or dist < best_d:
                best_d = dist
                best = Point2((float(opos.x), float(opos.y)))
        return best

    def _snapshot_spellcaster_spacing(self, unit: Any) -> None:
        """Freeze nearest-enemy point and energy for low-energy caster spacing."""
        tag = getattr(unit, "tag", None)
        if tag is None:
            return
        tag = int(tag)
        # Casters always snapshot. Swarm Hosts snapshot too, so a host that is
        # on cooldown can be paid for the same open-space move.
        if not self._is_backline_spellcaster(unit) and not self._is_swarm_host(unit):
            self._pre_action_nearest_enemy.pop(tag, None)
            self._pre_action_dist_to_nearest_enemy.pop(tag, None)
            self._pre_action_spell_energy.pop(tag, None)
            return
        if not self._is_backline_spellcaster(unit):
            self._pre_action_spell_energy.pop(tag, None)
            nearest = self._nearest_hostile_mobile_point(unit)
            if nearest is None:
                self._pre_action_nearest_enemy.pop(tag, None)
                self._pre_action_dist_to_nearest_enemy.pop(tag, None)
                return
            self._pre_action_nearest_enemy[tag] = nearest
            self._pre_action_dist_to_nearest_enemy[tag] = self._dist_to_point(
                unit, nearest
            )
            return
        energy = self._safe_energy(unit)
        if energy is None:
            self._pre_action_spell_energy.pop(tag, None)
        else:
            self._pre_action_spell_energy[tag] = energy
        nearest = self._nearest_hostile_mobile_point(unit)
        if nearest is None:
            self._pre_action_nearest_enemy.pop(tag, None)
            self._pre_action_dist_to_nearest_enemy.pop(tag, None)
            return
        self._pre_action_nearest_enemy[tag] = nearest
        self._pre_action_dist_to_nearest_enemy[tag] = self._dist_to_point(unit, nearest)

    def _add_conceal_shaping(self, amount: float) -> None:
        try:
            prev = float(getattr(self, "_step_conceal_shaping", 0.0) or 0.0)
        except (TypeError, ValueError):
            prev = 0.0
        self._step_conceal_shaping = prev + float(amount)

    def _credit_snipe_target_value(self, target: Any) -> None:
        """Credit a fresh ``snipe`` by the target's normalised resource cost.

        Higher-cost enemies pay more, in the same units as combat value
        (``minerals + vespene_weight × vespene``, divided by ``economic_value_norm``).
        Each enemy tag pays at most once per episode, so re-sniping the same
        body after the cooldown does not farm the bonus. The coefficient
        ``snipe_value_reward`` is applied when the step reward is summed.
        """
        r_conf = self.training_conf.get("rewards", {})
        try:
            coeff = float(r_conf.get("snipe_value_reward", 0.0) or 0.0)
        except (TypeError, ValueError):
            coeff = 0.0
        if coeff == 0.0 or target is None:
            return
        tag = getattr(target, "tag", None)
        if tag is None:
            return
        tag = int(tag)
        paid = getattr(self, "_snipe_value_paid_tags", None)
        if not isinstance(paid, set):
            paid = set()
            self._snipe_value_paid_tags = paid
        if tag in paid:
            return
        econ = EconomicRewardConfig.from_rewards_conf(r_conf)
        value = unit_economic_value(
            target,
            vespene_weight=econ.vespene_weight,
            value_norm=econ.value_norm,
        )
        if value <= 0.0:
            return
        paid.add(tag)
        try:
            prev = float(getattr(self, "_step_snipe_target_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            prev = 0.0
        self._step_snipe_target_value = prev + value

    def _credit_ambush_command(self, unit: Any, *, conceal: bool, r_conf: dict) -> None:
        """Widow Mine / Lurker: burrow pays per enemy in range; empty unburrow pays once.

        Burrowing with enemies inside attack range pays
        ``ambush_burrow_reward × count``. Unburrowing with nobody in that range
        pays ``ambush_unburrow_reward`` once, then stays quiet until the unit
        plants on at least one enemy again. Toggling in an empty field therefore
        pays a single correction, not a loop.
        """
        tag = getattr(unit, "tag", None)
        if tag is None:
            return
        tag = int(tag)
        disarmed = getattr(self, "_ambush_unburrow_disarmed", None)
        if not isinstance(disarmed, set):
            disarmed = set()
            self._ambush_unburrow_disarmed = disarmed
        if conceal:
            if self._ambush_is_burrowed(unit):
                return
            count = self._hostiles_in_ambush_range(unit)
            try:
                per_enemy = float(r_conf.get("ambush_burrow_reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                per_enemy = 0.0
            if count > 0 and per_enemy != 0.0:
                self._add_conceal_shaping(per_enemy * count)
                disarmed.discard(tag)
            return
        if not self._ambush_is_burrowed(unit):
            return
        if self._hostiles_in_ambush_range(unit) > 0:
            return
        if tag in disarmed:
            return
        try:
            bonus = float(r_conf.get("ambush_unburrow_reward", 0.0) or 0.0)
        except (TypeError, ValueError):
            bonus = 0.0
        if bonus == 0.0:
            return
        disarmed.add(tag)
        self._add_conceal_shaping(bonus)

    def _credit_conceal_command(self, unit: Any, *, conceal: bool) -> None:
        """Shape cloak / burrow at the moment the order is issued.

        Units at or below 10% health are paid ``low_hp_conceal_reward`` for
        cloaking or burrowing outside enemy detector range, and
        ``low_hp_reveal_reward`` for revealing while a detector covers them.
        Widow Mines and Lurkers skip that rule and use ambush credit instead.
        """
        if unit is None:
            return
        r_conf = self.training_conf.get("rewards", {}) or {}
        if self._is_ambush_unit(unit):
            self._credit_ambush_command(unit, conceal=conceal, r_conf=r_conf)
            return
        low_hp_limit = _LOW_HP_CONCEAL_FRACTION
        try:
            low_hp_limit = float(r_conf.get("low_hp_conceal_fraction", low_hp_limit))
        except (TypeError, ValueError):
            low_hp_limit = _LOW_HP_CONCEAL_FRACTION
        if self._health_fraction(unit) > low_hp_limit + 1e-6:
            return
        covered = self._enemy_detector_covers(unit)
        if conceal:
            if self._unit_is_concealed(unit) or covered:
                return
            try:
                bonus = float(r_conf.get("low_hp_conceal_reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                bonus = 0.0
        else:
            if not self._unit_is_concealed(unit) or not covered:
                return
            try:
                bonus = float(r_conf.get("low_hp_reveal_reward", 0.0) or 0.0)
            except (TypeError, ValueError):
                bonus = 0.0
        if bonus != 0.0:
            self._add_conceal_shaping(bonus)

    def calculate_spellcaster_retreat_reward(self, unit_tag: int) -> float:
        """Pay a backline unit for opening space when it cannot act.

        Ghost, High Templar, Infestor, and Viper qualify when pre-action energy
        is below the cheapest offensive spell. A Swarm Host qualifies when its
        locust launch was on cooldown. Moving away from the frozen nearest
        enemy pays ``spellcaster_retreat_scale × Δdist``, and only for the part
        of that gap that is still inside 10. Standing still, moving closer, or
        already being 10 away pays nothing.
        """
        if unit_tag is None:
            return 0.0
        r_conf = self.training_conf.get("rewards", {}) or {}
        try:
            scale = float(r_conf.get("spellcaster_retreat_scale", 0.0) or 0.0)
        except (TypeError, ValueError):
            scale = 0.0
        if scale == 0.0:
            return 0.0
        try:
            cap = float(
                r_conf.get("spellcaster_retreat_distance", _SPELLCASTER_RETREAT_DISTANCE)
            )
        except (TypeError, ValueError):
            cap = _SPELLCASTER_RETREAT_DISTANCE
        if cap <= 0.0:
            return 0.0

        unit = self.all_units.find_by_tag(unit_tag)
        if unit is None:
            return 0.0
        if self._is_backline_spellcaster(unit):
            floor = self._spellcaster_energy_floor(getattr(unit, "type_id", None))
            if floor <= 0.0:
                return 0.0
            energy = self._pre_action_spell_energy.get(unit_tag)
            if energy is None or float(energy) >= floor:
                return 0.0
        elif self._is_swarm_host(unit):
            if not self._swarm_host_on_cooldown(unit):
                return 0.0
        else:
            return 0.0
        prev_dist = self._pre_action_dist_to_nearest_enemy.get(unit_tag)
        frozen = self._pre_action_nearest_enemy.get(unit_tag)
        if prev_dist is None or frozen is None:
            return 0.0
        curr_dist = self._dist_to_point(unit, frozen)
        # Only the slice of the move that is still inside the cap counts.
        opened = min(float(curr_dist), cap) - min(float(prev_dist), cap)
        if opened <= 1e-3:
            return 0.0
        return scale * opened

    def _detection_step_reward(
        self,
        enemy_units: Iterable,
        own_units: Iterable,
        own_structs: Iterable,
        r_conf: dict,
        econ: EconomicRewardConfig,
    ) -> float:
        """Pay once per concealed hostile when it first enters an allied detector disc.

        The payout is ``detection_reward ×`` that unit's economic value, so two
        Dark Templar pay twice one, and a Banshee pays more than a Zergling.
        Paying the full value on every later step made keeping them alive worth
        more than killing them or winning (a Dark Templar is ~3 per step versus
        a win of 15), so the policy stalled the zone clear and timed out as a
        loss. Visible hostiles never count. The detector has to actually cover
        them.
        """
        detect_scale = float(r_conf.get("detection_reward", 0.0))
        if detect_scale == 0.0:
            return 0.0
        paid = getattr(self, "_detection_paid_tags", None)
        if not isinstance(paid, set):
            paid = set()
            self._detection_paid_tags = paid
        reward = 0.0
        for unit in enemy_units or []:
            if unit is None or getattr(unit, "is_structure", False) is True:
                continue
            if not self._unit_is_concealed(unit):
                continue
            if not self._allied_detector_covers(unit, own_units, own_structs):
                continue
            tag = getattr(unit, "tag", None)
            if tag is None or tag in paid:
                continue
            paid.add(tag)
            reward += detect_scale * unit_economic_value(
                unit,
                vespene_weight=econ.vespene_weight,
                value_norm=econ.value_norm,
            )
        return reward

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

        Enemy-value credit for both roles is scaled by distance to the role
        objective (full inside the inner radius, fading to
        ``objective_credit_min_scale``). Own-loss penalties are never scaled.
        """
        self._step_cooldown_retreat_reward = 0.0
        self._step_spellcaster_retreat = 0.0
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id

        reward = 0.0
        r_conf = self.training_conf.get("rewards", {})

        own_units, enemy_units, own_structs, enemy_structs = self._perspective_unit_groups(
            p_id
        )
        own_units, enemy_units, neural_suspended = apply_neural_reward_buckets(
            own_units, enemy_units, p_id
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
        
        # 2. Combat Rewards — resource-weighted economic trades
        #
        # Value flows, not hit points: damage pays the *fraction of a unit's
        # economic value* it removed, so gutting a 300/200 Thor is worth far more
        # than gutting a 25/0 Zergling. A death pays only the value still
        # standing, so per-step damage credit plus the death credit sum to exactly
        # the unit's price — no flat kill bonus double-counting.
        mobile_enemies_for_melee = [u for u in enemy_units if not u.is_structure]
        squad_is_melee = self._squad_has_melee_unit(squad_tags, mobile_enemies_for_melee)
        econ = EconomicRewardConfig.from_rewards_conf(r_conf)
        metrics = dict(EMPTY_ECONOMIC_METRICS)

        # Pylon damage — same economic units as army trades. Full destroy of the
        # enemy pylon pays ``pylon_value_scale × enemy_spawn_army_value ×
        # attacker_enemy_value_reward`` (frozen at prime). Own-pylon hits use
        # ``pylon_own_value_penalty`` (× defender scale when defending).
        army_ref = getattr(self, "_episode_army_value", None) or {}
        pylon_max = getattr(self, "_episode_pylon_max_vitality", None) or {}
        enemy_delta = (
            self.prev_pylon_health[enemy_global_idx] - curr_pylon_hp_map[enemy_global_idx]
        )
        if enemy_delta > 0.0:
            max_vit = float(pylon_max.get(enemy_global_idx, 0.0) or 0.0)
            if max_vit <= 1e-9:
                max_vit = float(self.prev_pylon_health[enemy_global_idx])
            reward += pylon_fraction_credit(
                enemy_delta,
                max_vitality=max_vit,
                reference_army_value=float(army_ref.get(enemy_global_idx, 0.0) or 0.0),
                value_scale=econ.pylon_value_scale,
                coefficient=econ.attacker_enemy_value_reward,
            )

        own_delta = (
            self.prev_pylon_health[own_global_idx] - curr_pylon_hp_map[own_global_idx]
        )
        if own_delta > 0.0:
            max_vit = float(pylon_max.get(own_global_idx, 0.0) or 0.0)
            if max_vit <= 1e-9:
                max_vit = float(self.prev_pylon_health[own_global_idx])
            own_coeff = float(econ.pylon_own_value_penalty)
            if self._squad_is_defend(squad_tags, self.unit_command_map):
                own_coeff *= float(econ.defender_pylon_penalty_scale)
            reward -= pylon_fraction_credit(
                own_delta,
                max_vitality=max_vit,
                reference_army_value=float(army_ref.get(own_global_idx, 0.0) or 0.0),
                value_scale=econ.pylon_value_scale,
                coefficient=own_coeff,
            )

        # Update global health tracking
        self.prev_pylon_health[1] = p1_pylon_hp
        self.prev_pylon_health[2] = p2_pylon_hp
        squad_is_defend = self._squad_is_defend(squad_tags, self.unit_command_map)

        # Permanent army only (timed summons carry no purchased value). Shots
        # *from* Interceptors still count: they move permanent units' value,
        # which is what this ledger tracks. Only Interceptor launch / recall /
        # despawn is ignored, so it cannot read as free damage or a free kill.
        curr_enemy_values = build_value_snapshot(
            enemy_units,
            config=econ,
            include_filter=counts_for_combat_hp_reward,
        )
        curr_own_values = build_value_snapshot(
            own_units,
            config=econ,
            include_filter=counts_for_combat_hp_reward,
            role_for_tag=lambda tag: self.unit_command_map.get(tag, ROLE_ATTACK),
        )
        prev_enemy_values = self.prev_enemy_value_snapshot.get(p_id) or {}
        prev_own_values = self.prev_own_value_snapshot.get(p_id) or {}

        # Both roles scale enemy-value credit by distance to the role objective
        # (attack: enemy pylon; defend: own pylon). Far fights still pay a
        # floor (``objective_credit_min_scale``) instead of zero.
        pos_scale = self._enemy_value_position_scale(
            squad_is_defend=squad_is_defend,
            squad_tags=squad_tags,
            own_pylon_pos=own_pylon_pos,
            enemy_objective=target_pos,
            r_conf=r_conf,
        )
        enemy_flow = value_delta(
            prev_enemy_values, curr_enemy_values, position_scale=pos_scale
        )
        enemy_coeff = econ.scaled_enemy_value_reward(
            econ.defender_enemy_value_reward
            if squad_is_defend
            else econ.attacker_enemy_value_reward,
            squad_is_melee=squad_is_melee,
        )
        reward += enemy_coeff * enemy_flow.lost
        metrics["economic_damage_dealt"] = enemy_flow.damaged
        metrics["economic_enemy_killed"] = enemy_flow.destroyed

        # Own attrition is never zone-gated, and is split by each lost unit's own
        # role: attackers pay a deliberately smaller penalty so a favourable (or
        # simply necessary) trade into the pylon stays learnable, while defenders
        # pay heavily for losing army they were supposed to preserve.
        own_baseline_ready = bool(prev_own_values)
        for role, coeff_base, metric in (
            (ROLE_ATTACK, econ.attacker_own_value_penalty, "economic_attacker_loss"),
            (ROLE_DEFEND, econ.defender_own_value_penalty, "economic_defender_loss"),
        ):
            flow = value_delta(
                prev_own_values,
                curr_own_values,
                role_filter=lambda unit_role, want=role: unit_role == want,
                ignore_missing_tags=neural_suspended,
            )
            metrics[metric] = flow.lost
            if not own_baseline_ready:
                continue
            coeff = econ.scaled_own_value_penalty(
                coeff_base, squad_is_melee=squad_is_melee
            )
            reward -= coeff * flow.lost

        self.prev_enemy_value_snapshot[p_id] = curr_enemy_values
        # prev_own_value_snapshot is written after the healing section below,
        # which needs the pre-step vitality to measure restored value.

        curr_enemy_snap = self._combat_snapshot(
            enemy_units, include_filter=counts_for_combat_hp_reward
        )
        self.prev_enemy_units_health[p_id] = self._mobile_combat_hp_sum(
            enemy_units, include_filter=counts_for_combat_hp_reward
        )
        self.prev_enemy_unit_count[p_id] = self._mobile_combat_count(
            enemy_units, include_filter=counts_for_combat_hp_reward
        )
        self.prev_enemy_combat_snapshot[p_id] = curr_enemy_snap

        curr_own_hp = self._mobile_combat_hp_sum(
            own_units, include_filter=counts_for_combat_hp_reward
        )
        # cast_tags: fresh-cast-only, gates heal_cast_reward (see
        # _issue_heal_at_point's continuing_same_patient comment).
        # order_tags: fresh OR continuing, gates the economic heal reward so a
        # multi-step Medivac beam / SCV repair keeps earning credit on every step
        # it actually ticks vitality, not just the step that started the lock
        # (which usually has not ticked any vitality yet).
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
        # Economic healing credit is double-gated: a live heal order from this
        # squad AND vitality actually restored. Restoring 10% of a 400/300
        # Battlecruiser pays 10% of its price; the same 10% on an SCV pays almost
        # nothing, which is exactly the trade a real player makes.
        own_heal_flow = value_delta(prev_own_values, curr_own_values)
        if healing_active and own_baseline_ready and own_heal_flow.restored > 0.0:
            reward += econ.heal_value_reward * own_heal_flow.restored
            metrics["economic_healing"] = own_heal_flow.restored
        if isinstance(cast_tags, set):
            cast_tags.clear()
        if isinstance(order_tags, set):
            order_tags.clear()
        attempt_tags = getattr(self, "_heal_attempt_tags", None)
        if isinstance(attempt_tags, set):
            attempt_tags.clear()
        self._clear_command_attempt_tags()
        self.prev_own_units_health[p_id] = curr_own_hp
        self.prev_own_value_snapshot[p_id] = curr_own_values
        self.prev_own_unit_count[p_id] = self._mobile_combat_count(
            own_units, include_filter=counts_for_combat_hp_reward
        )

        aoe_land_coeff = float(r_conf.get("aoe_land_reward", 0.0))
        aoe_land_value = float(getattr(self, "_step_aoe_land_value", 0.0) or 0.0)
        self._step_aoe_land_value = 0.0
        if aoe_land_coeff != 0.0 and aoe_land_value > 0.0:
            reward += aoe_land_coeff * aoe_land_value
        metrics["aoe_land_value"] = aoe_land_value

        snipe_value = float(getattr(self, "_step_snipe_target_value", 0.0) or 0.0)
        self._step_snipe_target_value = 0.0
        try:
            snipe_coeff = float(r_conf.get("snipe_value_reward", 0.0) or 0.0)
        except (TypeError, ValueError):
            snipe_coeff = 0.0
        if snipe_coeff != 0.0 and snipe_value > 0.0:
            reward += snipe_coeff * snipe_value
        metrics["snipe_target_value"] = snipe_value

        conceal = float(getattr(self, "_step_conceal_shaping", 0.0) or 0.0)
        self._step_conceal_shaping = 0.0
        if conceal != 0.0:
            reward += conceal
        metrics["conceal_shaping"] = conceal

        detection = self._detection_step_reward(
            enemy_units, own_units, own_structs, r_conf, econ
        )
        if detection != 0.0:
            reward += detection
        metrics["detection_reward"] = detection

        # 3. Progress Rewards — attacker squad Center-of-Mass distance shaping
        #
        # When an attacker unit is active: reward the CoM of ALL attacker units
        # (same player) closing in on the target pylon.
        # Defenders use per-unit zone penalty (section 3c) instead of CoM shaping.
        active_unit_role = self.unit_command_map.get(self.active_unit_tag, "attack") if self.active_unit_tag else "attack"

        if active_unit_role == "attack":
            # Casters always stay out of this center of mass. A Swarm Host stays
            # out only while locusts are on cooldown; a ready host is pulled in
            # so it can walk to launch range.
            squad_units = [
                u
                for u in own_units
                if not u.is_structure
                and self.unit_command_map.get(u.tag, "attack") == "attack"
                and not self._is_backline_spellcaster(u)
                and not (
                    self._is_swarm_host(u) and self._swarm_host_on_cooldown(u)
                )
            ]
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
        zone_total = 0.0
        for tag in squad_tags:
            self.active_unit_tag = tag
            retreat = self.calculate_cooldown_retreat_reward(tag, mobile_enemies)
            self._step_cooldown_retreat_reward = (
                float(getattr(self, "_step_cooldown_retreat_reward", 0.0)) + float(retreat)
            )
            reward += retreat
            spacing = self.calculate_spellcaster_retreat_reward(tag)
            self._step_spellcaster_retreat = (
                float(getattr(self, "_step_spellcaster_retreat", 0.0)) + float(spacing)
            )
            reward += spacing
            zone = self.calculate_defender_zone_penalty(tag, own_pylon_pos)
            zone_total += float(zone)
            reward += zone
        self.active_unit_tag = saved_active
        metrics["defender_zone_shaping"] = zone_total
        metrics["spellcaster_retreat"] = float(
            getattr(self, "_step_spellcaster_retreat", 0.0) or 0.0
        )
        self._step_economic_metrics = metrics

        # Both roles pay the step penalty. The old per-step defender_step_reward is
        # gone: paying a defender simply for existing is farmable by hiding in a
        # corner for the whole episode. Army preservation is now rewarded once, at
        # the end, by terminal_retained_value_bonus.
        if squad_tags:
            reward += r_conf.get("step_penalty", -0.05)

        return reward

    def pop_economic_metrics(self) -> Dict[str, float]:
        """Take this step's economic telemetry (see ``ECONOMIC_METRIC_KEYS``)."""
        metrics = dict(getattr(self, "_step_economic_metrics", None) or EMPTY_ECONOMIC_METRICS)
        self._step_economic_metrics = dict(EMPTY_ECONOMIC_METRICS)
        return metrics

    def retained_defender_value_fraction(
        self, perspective_player_id: Optional[int] = None
    ) -> float:
        """Share of this player's *defender* army value still standing, in [0, 1].

        Read at episode end for the terminal preservation bonus. Uses the value
        snapshot from the last reward step, so it reflects the same ledger the
        per-step penalties used.
        """
        p_id = perspective_player_id if perspective_player_id is not None else self.player_id
        snapshot = self.prev_own_value_snapshot.get(int(p_id)) or {}
        return retained_value_fraction(
            snapshot, role_filter=lambda role: role == ROLE_DEFEND
        )

    def terminal_retained_value_bonus(
        self,
        squad_tags: List[int],
        perspective_player_id: Optional[int] = None,
    ) -> float:
        """Terminal defender bonus proportional to army value preserved.

        Zero for attacker squads (attackers are paid for pressure, not survival)
        and zero for a wiped defender.
        """
        if not self._squad_is_defend(squad_tags, self.unit_command_map):
            return 0.0
        econ = EconomicRewardConfig.from_rewards_conf(
            self.training_conf.get("rewards", {})
        )
        if econ.defender_retained_value_bonus == 0.0:
            return 0.0
        return econ.defender_retained_value_bonus * self.retained_defender_value_fraction(
            perspective_player_id
        )

    def calculate_reward(self, perspective_player_id: Optional[int] = None) -> float:
        """Single-unit wrapper around :meth:`calculate_squad_step_reward`."""
        squad_tags = [self.active_unit_tag] if self.active_unit_tag is not None else []
        return self.calculate_squad_step_reward(squad_tags, perspective_player_id)
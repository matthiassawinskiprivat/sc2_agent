"""Resource-weighted, role-specific combat rewards for the CTF squad curriculum.

StarCraft II is an economy game: killing a 300/200 Thor is worth far more than
killing a 25/0 Zergling, and the flat hit-point shaping this module replaces paid
both the same per point of health. Here every unit carries an **economic value**

    value = minerals + vespene_weight * vespene          (vespene_weight ~ 1.5)

normalised by a configurable constant so episode returns stay in a numerically
comfortable range. Damage is credited as the *fraction of that value destroyed*,
so a shot that removes 10% of a unit's vitality pays 10% of its value regardless
of whether the unit has 45 or 400 hit points.

A death pays only the value that was **still standing**, so the flat kill bonus
can no longer double-count damage that was already rewarded step by step.

Roles are asymmetric on purpose:

* **Attackers** get a strong enemy-value reward and a deliberately *smaller* own
  loss penalty, so a favourable-but-costly trade (or a necessary suicide into the
  pylon) stays learnable.
* **Defenders** get a strong own-value preservation penalty, enemy-value credit
  only for fighting inside their pylon zone, a heavier pylon-damage penalty, and a
  **terminal** retained-army-value bonus — not a per-step survival payout, which
  a defender could farm by hiding in a corner for the whole episode.

Vitality means health + shields. Temporary bodies (Interceptors, Locusts,
Broodlings, hallucinations) are excluded by the caller's ``include_filter`` so a
Carrier launching Interceptors does not look like free army value appearing and
disappearing; embarked passengers are included so a dropship death pays for its
cargo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

DEFAULT_VESPENE_WEIGHT = 1.5
DEFAULT_VALUE_NORM = 100.0

# Fallback used when a unit type is missing from the cost table. Small but
# nonzero: a silent 0 would make an unmodelled unit free to lose and worthless
# to kill, which is a much more damaging failure than a slightly wrong price.
FALLBACK_MINERALS = 50
FALLBACK_VESPENE = 0

ROLE_ATTACK = "attack"
ROLE_DEFEND = "defend"


@dataclass(frozen=True)
class EconomicRewardConfig:
    """Role coefficients and value scaling, all overridable from level YAML."""

    vespene_weight: float = DEFAULT_VESPENE_WEIGHT
    value_norm: float = DEFAULT_VALUE_NORM
    attacker_enemy_value_reward: float = 1.0
    attacker_own_value_penalty: float = 0.35
    defender_enemy_value_reward: float = 0.8
    defender_own_value_penalty: float = 1.2
    defender_pylon_penalty_scale: float = 1.5
    defender_retained_value_bonus: float = 3.0
    heal_value_reward: float = 1.0
    melee_multiplier: float = 0.5

    @classmethod
    def from_rewards_conf(cls, r_conf: Optional[dict]) -> "EconomicRewardConfig":
        conf = r_conf or {}

        def _f(key: str, default: float) -> float:
            try:
                return float(conf.get(key, default))
            except (TypeError, ValueError):
                return default

        norm = _f("economic_value_norm", DEFAULT_VALUE_NORM)
        return cls(
            vespene_weight=_f("economic_vespene_weight", DEFAULT_VESPENE_WEIGHT),
            value_norm=norm if abs(norm) > 1e-6 else DEFAULT_VALUE_NORM,
            attacker_enemy_value_reward=_f("attacker_enemy_value_reward", 1.0),
            attacker_own_value_penalty=_f("attacker_own_value_penalty", 0.35),
            defender_enemy_value_reward=_f("defender_enemy_value_reward", 0.8),
            defender_own_value_penalty=_f("defender_own_value_penalty", 1.2),
            defender_pylon_penalty_scale=_f("defender_pylon_penalty_scale", 1.5),
            defender_retained_value_bonus=_f("defender_retained_value_bonus", 3.0),
            heal_value_reward=_f("heal_value_reward", 1.0),
            melee_multiplier=_f("melee_multiplier", 0.5),
        )

    def scaled_enemy_value_reward(self, base: float, *, squad_is_melee: bool) -> float:
        """Melee squads must close the gap to deal damage; keep the old scaling."""
        if not squad_is_melee:
            return base
        return base * (self.melee_multiplier + 1.0)

    def scaled_own_value_penalty(self, base: float, *, squad_is_melee: bool) -> float:
        if not squad_is_melee:
            return base
        return base * self.melee_multiplier


@dataclass(frozen=True)
class UnitValue:
    """One permanent unit's economic and vitality state at a point in time."""

    type_id: int
    value: float            # normalised economic value of the whole unit
    max_vitality: float     # health_max + shield_max
    vitality: float         # current health + shield
    position: Optional[Any]
    role: str

    @property
    def vitality_fraction(self) -> float:
        return self.fraction_of(self.max_vitality)

    def fraction_of(self, max_vitality: float) -> float:
        """Vitality as a share of *max_vitality*, clamped to ``[0, 1]``.

        Deltas are always measured against the **pre-step** maximum so a frame
        that reports no ``health_max`` (mocks, or a proto without the field) can
        never make real damage read as zero.
        """
        if max_vitality <= 0.0:
            return 0.0
        return max(0.0, min(1.0, self.vitality / max_vitality))

    @property
    def standing_value(self) -> float:
        """Value still on the field (what a death would destroy right now)."""
        return self.value * self.vitality_fraction


@dataclass(frozen=True)
class ValueDelta:
    """Economic value moved between two snapshots of the same unit group."""

    damaged: float = 0.0     # value removed by damage on surviving units
    destroyed: float = 0.0   # value still standing on units that died
    restored: float = 0.0    # value healed back onto surviving units
    kills: int = 0

    @property
    def lost(self) -> float:
        return self.damaged + self.destroyed


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unit_cost(unit: Any) -> Tuple[int, int]:
    """``(minerals, vespene)`` for a unit, following transformed-mode aliases."""
    from train.envs.ctf.game_data import get_unit_cost

    type_id = getattr(unit, "type_id", None)
    raw = getattr(type_id, "value", type_id)
    try:
        minerals, vespene = get_unit_cost(int(raw))
    except (TypeError, ValueError):
        return FALLBACK_MINERALS, FALLBACK_VESPENE
    if minerals <= 0 and vespene <= 0:
        return FALLBACK_MINERALS, FALLBACK_VESPENE
    return minerals, vespene


def unit_economic_value(
    unit: Any,
    *,
    vespene_weight: float = DEFAULT_VESPENE_WEIGHT,
    value_norm: float = DEFAULT_VALUE_NORM,
) -> float:
    """Normalised ``minerals + vespene_weight * vespene`` for one unit."""
    minerals, vespene = unit_cost(unit)
    raw = float(minerals) + float(vespene_weight) * float(vespene)
    norm = float(value_norm) if abs(float(value_norm)) > 1e-6 else DEFAULT_VALUE_NORM
    return raw / norm


def _unit_vitality(unit: Any) -> Tuple[float, float]:
    health = _safe_float(getattr(unit, "health", 0.0))
    shield = _safe_float(getattr(unit, "shield", 0.0))
    health_max = _safe_float(getattr(unit, "health_max", 0.0))
    shield_max = _safe_float(getattr(unit, "shield_max", 0.0))
    max_vit = health_max + shield_max
    if max_vit <= 0.0:
        # Mocks / unknown proto: treat current vitality as full so a survivor
        # never reads as "damaged to 0%" on the first snapshot.
        max_vit = max(health + shield, 1.0)
    return health + shield, max_vit


def build_value_snapshot(
    units: Iterable[Any],
    *,
    config: EconomicRewardConfig,
    include_filter: Optional[Callable[[Any], bool]] = None,
    role_for_tag: Optional[Callable[[int], str]] = None,
) -> Dict[int, UnitValue]:
    """Per-tag economic/vitality snapshot, including embarked passengers.

    Cargo uses the transport's position so defender-zone gating cannot be dodged
    by loading units into a dropship.
    """
    snapshot: Dict[int, UnitValue] = {}

    def _add(unit: Any, position: Any) -> None:
        tag = getattr(unit, "tag", None)
        if tag is None or int(tag) in snapshot:
            return
        if include_filter is not None and not include_filter(unit):
            return
        vitality, max_vitality = _unit_vitality(unit)
        role = ROLE_ATTACK
        if role_for_tag is not None:
            role = str(role_for_tag(int(tag)) or ROLE_ATTACK).lower()
        type_raw = getattr(unit, "type_id", None)
        snapshot[int(tag)] = UnitValue(
            type_id=int(getattr(type_raw, "value", 0) or 0),
            value=unit_economic_value(
                unit,
                vespene_weight=config.vespene_weight,
                value_norm=config.value_norm,
            ),
            max_vitality=max_vitality,
            vitality=vitality,
            position=position,
            role=role,
        )

    for unit in units or []:
        if unit is None:
            continue
        if include_filter is None and getattr(unit, "is_structure", False):
            continue
        carrier_pos = getattr(unit, "position", None)
        _add(unit, carrier_pos)
        for passenger in getattr(unit, "passengers", None) or []:
            if passenger is not None:
                _add(passenger, carrier_pos)
    return snapshot


def value_delta(
    previous: Dict[int, UnitValue],
    current: Dict[int, UnitValue],
    *,
    position_filter: Optional[Callable[[Any], bool]] = None,
    role_filter: Optional[Callable[[str], bool]] = None,
) -> ValueDelta:
    """Damaged / destroyed / restored economic value between two snapshots.

    ``position_filter`` gates credit by where the change happened (defender zone).
    ``role_filter`` gates by the unit's own high-level role (attacker vs defender
    loss accounting). A unit that dies contributes only the value it still had,
    so per-step damage credit and the death credit sum to exactly its full value.
    """
    damaged = 0.0
    destroyed = 0.0
    restored = 0.0
    kills = 0
    for tag, before in previous.items():
        if role_filter is not None and not role_filter(before.role):
            continue
        after = current.get(tag)
        position = before.position if after is None else (
            after.position if after.position is not None else before.position
        )
        if position_filter is not None and not position_filter(position):
            continue
        if after is None:
            destroyed += before.standing_value
            kills += 1
            continue
        anchor = before.max_vitality
        delta_fraction = after.fraction_of(anchor) - before.fraction_of(anchor)
        if delta_fraction < 0.0:
            damaged += -delta_fraction * before.value
        elif delta_fraction > 0.0:
            restored += delta_fraction * before.value
    return ValueDelta(
        damaged=damaged, destroyed=destroyed, restored=restored, kills=kills
    )


def retained_value_fraction(
    snapshot: Dict[int, UnitValue],
    *,
    role_filter: Optional[Callable[[str], bool]] = None,
) -> float:
    """Share of the snapshot's total value still standing, in ``[0, 1]``.

    Used for the defender's terminal bonus. Returns ``0.0`` for an empty group so
    a wiped defender earns nothing.
    """
    total = 0.0
    standing = 0.0
    for entry in snapshot.values():
        if role_filter is not None and not role_filter(entry.role):
            continue
        total += entry.value
        standing += entry.standing_value
    if total <= 0.0:
        return 0.0
    return max(0.0, min(1.0, standing / total))


def trade_efficiency(enemy_value_destroyed: float, own_value_lost: float) -> float:
    """Share of traded value that came off the enemy, in ``[0, 1]``.

    ``0.5`` is an even trade; above ``0.5`` the squad is winning the exchange.
    Returns ``0.0`` when nothing was traded at all.
    """
    total = float(enemy_value_destroyed) + float(own_value_lost)
    if total <= 1e-9:
        return 0.0
    return float(enemy_value_destroyed) / total

"""Shared unit-level feature helpers used by observation builders.

Extracted from ObservationTransformer so squad and other builders can call them
without importing the full transformer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, FrozenSet

if TYPE_CHECKING:
    from sc2.position import Point2
    from sc2.unit import Unit


try:
    from sc2.ids.ability_id import AbilityId
    from sc2.ids.unit_typeid import UnitTypeId
    from sc2.ids.upgrade_id import UpgradeId
    _SC2_AVAILABLE = True
except ImportError:  # pragma: no cover — SC2 not installed in CI
    _SC2_AVAILABLE = False

# Short-reach combat units whose catalog/game range can exceed the generic
# melee threshold (1.5) but should still be treated as melee for CTF shaping
# and the squad ``is_melee`` scalar.
_FORCE_MELEE_UNIT_TYPES: FrozenSet = (
    frozenset({UnitTypeId.HELLIONTANK, UnitTypeId.ARCHON}) if _SC2_AVAILABLE else frozenset()
)

# Timed summons that expire on their own. Own-side kill_penalty / damage_penalty
# must ignore them so natural expiration is not treated as attrition.
_TEMPORARY_ATTRITION_UNIT_TYPES: FrozenSet = (
    frozenset(
        {
            UnitTypeId.BROODLING,
            UnitTypeId.LOCUSTMP,
            UnitTypeId.LOCUSTMPFLYING,
            UnitTypeId.DISRUPTORPHASED,
            UnitTypeId.INTERCEPTOR,
        }
    )
    if _SC2_AVAILABLE
    else frozenset()
)

# Swarm Hosts have no weapon; policy ``attack`` heads map to Spawn Locusts at
# the same world point (agent-faithful target, unit-correct ability).
_SPAWN_LOCUSTS_UNIT_TYPES: FrozenSet = (
    frozenset({UnitTypeId.SWARMHOSTMP, UnitTypeId.SWARMHOSTBURROWEDMP})
    if _SC2_AVAILABLE
    else frozenset()
)

# Disruptors have no weapon; policy ``attack`` heads map to Purification Nova at
# the same world point (same agent-faithful pattern as Swarm Host).
_PURIFICATION_NOVA_UNIT_TYPES: FrozenSet = (
    frozenset({UnitTypeId.DISRUPTOR}) if _SC2_AVAILABLE else frozenset()
)


def is_force_melee_unit(unit: "Unit") -> bool:
    """True for unit types always treated as melee regardless of weapon range."""
    return getattr(unit, "type_id", None) in _FORCE_MELEE_UNIT_TYPES


def is_temporary_attrition_unit(unit: "Unit") -> bool:
    """True for timed summons excluded from own kill/damage attrition penalties."""
    return getattr(unit, "type_id", None) in _TEMPORARY_ATTRITION_UNIT_TYPES


def uses_spawn_locusts_for_attack(unit: "Unit") -> bool:
    """True when policy attack should cast Spawn Locusts instead of ``unit.attack``."""
    return getattr(unit, "type_id", None) in _SPAWN_LOCUSTS_UNIT_TYPES


def uses_purification_nova_for_attack(unit: "Unit") -> bool:
    """True when policy attack should cast Purification Nova instead of ``unit.attack``."""
    return getattr(unit, "type_id", None) in _PURIFICATION_NOVA_UNIT_TYPES


def issue_attack_at_point(bot: Any, unit: "Unit", target: "Point2") -> None:
    """Issue an offensive order at *target*.

    Swarm Hosts (and burrowed) cast ``EFFECT_SPAWNLOCUSTS`` at the same point;
    Disruptors cast ``EFFECT_PURIFICATIONNOVA`` at the same point;
    all other units use ``unit.attack(target)``.
    """
    if uses_spawn_locusts_for_attack(unit):
        bot.do(unit(AbilityId.EFFECT_SPAWNLOCUSTS, target))
    elif uses_purification_nova_for_attack(unit):
        bot.do(unit(AbilityId.EFFECT_PURIFICATIONNOVA, target))
    else:
        bot.do(unit.attack(target))


def weapon_range_upgrade_deltas(unit: "Unit", upgrades: frozenset) -> tuple[float, float]:
    """Return (ground_range_delta, air_range_delta) from researched upgrades.

    python-sc2 ``Unit.ground_range`` / ``air_range`` read static GameData and do
    not include these bonuses.  Pass ``bot.state.upgrades`` (own player) or an
    inferred frozenset for the enemy.

    Returns (0.0, 0.0) when no relevant upgrade is present.
    """
    if not upgrades or not _SC2_AVAILABLE:
        return 0.0, 0.0
    uid = unit.type_id
    if uid == UnitTypeId.COLOSSUS and UpgradeId.EXTENDEDTHERMALLANCE in upgrades:
        return 3.0, 0.0
    if uid == UnitTypeId.HYDRALISK and UpgradeId.EVOLVEGROOVEDSPINES in upgrades:
        return 1.0, 1.0
    if uid == UnitTypeId.PHOENIX and UpgradeId.PHOENIXRANGEUPGRADE in upgrades:
        return 0.0, 2.0
    if (
        uid in {UnitTypeId.PLANETARYFORTRESS, UnitTypeId.MISSILETURRET, UnitTypeId.AUTOTURRET}
        and UpgradeId.HISECAUTOTRACKING in upgrades
    ):
        return 1.0, 1.0
    return 0.0, 0.0

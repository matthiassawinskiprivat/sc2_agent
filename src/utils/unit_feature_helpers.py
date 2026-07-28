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
# Controllable mid-episode summons still join the squad pointer; they are only
# excluded from attrition accounting.
_TEMPORARY_ATTRITION_UNIT_TYPES: FrozenSet = (
    frozenset(
        {
            UnitTypeId.BROODLING,
            UnitTypeId.LOCUSTMP,
            UnitTypeId.LOCUSTMPFLYING,
            UnitTypeId.DISRUPTORPHASED,
            UnitTypeId.INTERCEPTOR,
            UnitTypeId.ADEPTPHASESHIFT,
            UnitTypeId.INFESTORTERRAN,
            UnitTypeId.INFESTORTERRANBURROWED,
            UnitTypeId.CHANGELING,
            UnitTypeId.CHANGELINGZEALOT,
            UnitTypeId.CHANGELINGMARINE,
            UnitTypeId.CHANGELINGMARINESHIELD,
            UnitTypeId.CHANGELINGZERGLING,
            UnitTypeId.CHANGELINGZERGLINGWINGS,
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

# Controllable Purification Nova ball (steer with move / attack→move).
_PURIFICATION_NOVA_BALL_TYPES: FrozenSet = (
    frozenset({UnitTypeId.DISRUPTORPHASED}) if _SC2_AVAILABLE else frozenset()
)

# Mid-episode controllable summons → preferred same-owner parent types.
# API exposes MOVE/ATTACK on all of these (including Interceptors). Structures
# like Auto-Turrets are excluded via ``is_structure`` during adoption.
_MIDGAME_SUMMON_PARENTS: dict = (
    {
        UnitTypeId.DISRUPTORPHASED: frozenset({UnitTypeId.DISRUPTOR}),
        UnitTypeId.LOCUSTMP: frozenset(
            {UnitTypeId.SWARMHOSTMP, UnitTypeId.SWARMHOSTBURROWEDMP}
        ),
        UnitTypeId.LOCUSTMPFLYING: frozenset(
            {UnitTypeId.SWARMHOSTMP, UnitTypeId.SWARMHOSTBURROWEDMP}
        ),
        UnitTypeId.BROODLING: frozenset({UnitTypeId.BROODLORD}),
        UnitTypeId.INTERCEPTOR: frozenset({UnitTypeId.CARRIER}),
        UnitTypeId.ADEPTPHASESHIFT: frozenset({UnitTypeId.ADEPT}),
        UnitTypeId.INFESTORTERRAN: frozenset(
            {UnitTypeId.INFESTOR, UnitTypeId.INFESTORBURROWED}
        ),
        UnitTypeId.INFESTORTERRANBURROWED: frozenset(
            {UnitTypeId.INFESTOR, UnitTypeId.INFESTORBURROWED}
        ),
        UnitTypeId.CHANGELING: frozenset(
            {UnitTypeId.OVERSEER, UnitTypeId.OVERSEERSIEGEMODE}
        ),
        UnitTypeId.CHANGELINGZEALOT: frozenset(
            {UnitTypeId.OVERSEER, UnitTypeId.OVERSEERSIEGEMODE}
        ),
        UnitTypeId.CHANGELINGMARINE: frozenset(
            {UnitTypeId.OVERSEER, UnitTypeId.OVERSEERSIEGEMODE}
        ),
        UnitTypeId.CHANGELINGMARINESHIELD: frozenset(
            {UnitTypeId.OVERSEER, UnitTypeId.OVERSEERSIEGEMODE}
        ),
        UnitTypeId.CHANGELINGZERGLING: frozenset(
            {UnitTypeId.OVERSEER, UnitTypeId.OVERSEERSIEGEMODE}
        ),
        UnitTypeId.CHANGELINGZERGLINGWINGS: frozenset(
            {UnitTypeId.OVERSEER, UnitTypeId.OVERSEERSIEGEMODE}
        ),
    }
    if _SC2_AVAILABLE
    else {}
)
_MIDGAME_SUMMON_TYPES: FrozenSet = (
    frozenset(_MIDGAME_SUMMON_PARENTS) if _SC2_AVAILABLE else frozenset()
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


def is_purification_nova_ball(unit: "Unit") -> bool:
    """True for the controllable Purification Nova ball (``DISRUPTORPHASED``)."""
    return getattr(unit, "type_id", None) in _PURIFICATION_NOVA_BALL_TYPES


def is_midgame_summon(unit: "Unit") -> bool:
    """True for controllable units spawned mid-episode (locust, broodling, …)."""
    return getattr(unit, "type_id", None) in _MIDGAME_SUMMON_TYPES


def adopt_midgame_summons_into_pools(
    all_units: Any,
    unit_pool_index: dict,
    unit_command_map: Any = None,
    unit_target_map: Any = None,
) -> None:
    """Attach unmapped mid-episode summons to their preferred parent's squad pool.

    Locusts, Broodlings, Interceptors, Nova balls, Adept shades, Infested Terrans,
    and Changelings are created mid-episode and are absent from spawn-time
    ``unit_pool_index``. Without adoption they never appear in squad pointer slots.
    Attrition rewards still ignore them via ``is_temporary_attrition_unit``.
    """
    if not unit_pool_index or not _MIDGAME_SUMMON_PARENTS:
        return

    for unit in all_units:
        if getattr(unit, "is_structure", False):
            continue
        type_id = getattr(unit, "type_id", None)
        preferred_parents = _MIDGAME_SUMMON_PARENTS.get(type_id)
        if not preferred_parents:
            continue
        tag = int(unit.tag)
        if tag in unit_pool_index:
            continue

        best_parent = None
        best_dist = float("inf")
        owner = getattr(unit, "owner_id", None)
        # Prefer a same-owner parent of the expected caster type.
        for other in all_units:
            if getattr(other, "owner_id", None) != owner:
                continue
            other_tag = int(other.tag)
            if other_tag not in unit_pool_index:
                continue
            if getattr(other, "type_id", None) not in preferred_parents:
                continue
            dist = float(unit.distance_to(other))
            if dist < best_dist:
                best_dist = dist
                best_parent = other
        # Fallback: nearest same-owner pooled unit if no preferred parent is found.
        if best_parent is None:
            for other in all_units:
                if getattr(other, "owner_id", None) != owner:
                    continue
                other_tag = int(other.tag)
                if other_tag not in unit_pool_index:
                    continue
                dist = float(unit.distance_to(other))
                if dist < best_dist:
                    best_dist = dist
                    best_parent = other
        if best_parent is None:
            continue

        parent_tag = int(best_parent.tag)
        unit_pool_index[tag] = unit_pool_index[parent_tag]
        if isinstance(unit_command_map, dict):
            unit_command_map[tag] = unit_command_map.get(parent_tag, "attack")
        if isinstance(unit_target_map, dict) and parent_tag in unit_target_map:
            unit_target_map[tag] = unit_target_map[parent_tag]


def adopt_purification_nova_into_pools(
    all_units: Any,
    unit_pool_index: dict,
    unit_command_map: Any = None,
    unit_target_map: Any = None,
) -> None:
    """Backward-compatible alias for ``adopt_midgame_summons_into_pools``."""
    adopt_midgame_summons_into_pools(
        all_units,
        unit_pool_index,
        unit_command_map=unit_command_map,
        unit_target_map=unit_target_map,
    )


def prune_dead_from_pool_maps(
    alive_tags: Any,
    unit_pool_index: dict,
    unit_command_map: Any = None,
    unit_target_map: Any = None,
) -> list:
    """Drop dead tags from pool membership maps (e.g. expired ``DISRUPTORPHASED``).

    ``squad_unit_tags`` already filters by alive tags, but leaving ghosts in
    ``unit_pool_index`` / command maps keeps stale anchors and map clutter.
    Returns the pruned tag list.
    """
    alive = {int(t) for t in alive_tags}
    dead = [int(t) for t in list(unit_pool_index) if int(t) not in alive]
    for tag in dead:
        unit_pool_index.pop(tag, None)
        if isinstance(unit_command_map, dict):
            unit_command_map.pop(tag, None)
        if isinstance(unit_target_map, dict):
            unit_target_map.pop(tag, None)
    return dead


def issue_attack_at_point(bot: Any, unit: "Unit", target: "Point2") -> None:
    """Issue an offensive order at *target*.

    Swarm Hosts (and burrowed) cast ``EFFECT_SPAWNLOCUSTS`` at the same point;
    Disruptors cast ``EFFECT_PURIFICATIONNOVA`` at the same point;
    Purification Nova balls ``move`` to the point (no weapon);
    all other units use ``unit.attack(target)``.
    """
    if uses_spawn_locusts_for_attack(unit):
        bot.do(unit(AbilityId.EFFECT_SPAWNLOCUSTS, target))
    elif uses_purification_nova_for_attack(unit):
        bot.do(unit(AbilityId.EFFECT_PURIFICATIONNOVA, target))
    elif is_purification_nova_ball(unit):
        bot.do(unit.move(target))
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

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


# Morph / burrow states that cannot move (independent of Tunneling Claws).
_ALWAYS_IMMOBILE_TYPES: FrozenSet = (
    frozenset(
        {
            UnitTypeId.SIEGETANKSIEGED,
            UnitTypeId.WIDOWMINEBURROWED,
            UnitTypeId.LURKERMPBURROWED,
            UnitTypeId.LURKERBURROWED,
            UnitTypeId.LIBERATORAG,
            UnitTypeId.OBSERVERSIEGEMODE,
            UnitTypeId.OVERSEERSIEGEMODE,
        }
    )
    if _SC2_AVAILABLE
    else frozenset()
)

_TUNNELING_CLAWS_MOBILE_BURROWED: FrozenSet = (
    frozenset({UnitTypeId.ROACHBURROWED, UnitTypeId.INFESTORBURROWED})
    if _SC2_AVAILABLE
    else frozenset()
)

_SWARM_HOST_BURROWED: FrozenSet = (
    frozenset({UnitTypeId.SWARMHOSTBURROWEDMP})
    if _SC2_AVAILABLE
    else frozenset()
)

# Personal cloak activation energy (Ghost / Banshee).
_CLOAK_ACTIVATION_ENERGY = 25.0
# Mothership cloak-field energy gate (approximation).
_MOTHERSHIP_CLOAK_ENERGY = 50.0


def unit_is_immobile(unit: "Unit", upgrades: frozenset | None = None) -> bool:
    """True when the unit cannot move in its current morph/burrow state."""
    if not _SC2_AVAILABLE or unit is None:
        return False
    upgrades = upgrades or frozenset()
    tid = getattr(unit, "type_id", None)
    if tid in _ALWAYS_IMMOBILE_TYPES:
        return True
    if tid in _SWARM_HOST_BURROWED:
        return False
    is_burrowed = bool(getattr(unit, "is_burrowed", False)) or (
        tid is not None and "BURROWED" in str(getattr(tid, "name", "")).upper()
    )
    if not is_burrowed:
        return False
    if tid in _TUNNELING_CLAWS_MOBILE_BURROWED and UpgradeId.TUNNELINGCLAWS in upgrades:
        return False
    # Burrowed Zerg (and Widow Mine handled above): immobile without claws.
    return True


def unit_can_cloak(unit: "Unit") -> bool:
    """True when the unit can activate personal/field cloak this step.

    Already-cloaked units return False. Energy-gated for Ghost/Banshee/Mothership.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    if bool(getattr(unit, "is_cloaked", False)):
        return False
    tid = getattr(unit, "type_id", None)
    energy = float(getattr(unit, "energy", 0.0) or 0.0)
    if tid in {UnitTypeId.GHOST, UnitTypeId.BANSHEE}:
        return energy >= _CLOAK_ACTIVATION_ENERGY
    if tid == UnitTypeId.MOTHERSHIP:
        return energy >= _MOTHERSHIP_CLOAK_ENERGY
    return False


def unit_can_repair(unit: "Unit") -> bool:
    """True for SCVs (mechanical repair); False for all other types."""
    if not _SC2_AVAILABLE or unit is None:
        return False
    return getattr(unit, "type_id", None) == UnitTypeId.SCV


def unit_can_heal(unit: "Unit") -> bool:
    """True when a non-SCV healer can cast heal this step (energy-gated).

    Medivac / Queen / Shield Battery. SCV uses ``unit_can_repair`` instead.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    energy = float(getattr(unit, "energy", 0.0) or 0.0)
    if tid == UnitTypeId.MEDIVAC:
        return energy > 5.0
    if tid == UnitTypeId.QUEEN:
        return energy >= 50.0
    if tid == UnitTypeId.SHIELDBATTERY:
        return energy > 3.0
    return False


_CARGO_SIZE_NORM_DIVISOR = 8.0


def unit_is_transport(unit: "Unit") -> bool:
    if unit is None:
        return False
    return int(getattr(unit, "cargo_max", 0) or 0) > 0


def unit_cargo_used_norm(unit: "Unit") -> float:
    cargo_max = int(getattr(unit, "cargo_max", 0) or 0)
    if cargo_max <= 0:
        return 0.0
    used = int(getattr(unit, "cargo_used", 0) or 0)
    return min(float(used) / float(cargo_max), 1.0)


def unit_cargo_free_norm(unit: "Unit") -> float:
    cargo_max = int(getattr(unit, "cargo_max", 0) or 0)
    if cargo_max <= 0:
        return 0.0
    used = int(getattr(unit, "cargo_used", 0) or 0)
    return max(float(cargo_max - used) / float(cargo_max), 0.0)


def unit_cargo_size_norm(unit: "Unit") -> float:
    """Embark cost for non-transports; 0 for transports / unloadable."""
    if unit is None or unit_is_transport(unit):
        return 0.0
    size = float(getattr(unit, "cargo_size", 0) or 0)
    if size <= 0:
        return 0.0
    return min(size / _CARGO_SIZE_NORM_DIVISOR, 1.0)


def unit_can_load(unit: "Unit") -> bool:
    """Transport with free space, or non-transport with positive cargo_size."""
    if unit is None:
        return False
    cargo_max = int(getattr(unit, "cargo_max", 0) or 0)
    if cargo_max > 0:
        return int(getattr(unit, "cargo_used", 0) or 0) < cargo_max
    return float(getattr(unit, "cargo_size", 0) or 0) > 0


def unit_can_unload(unit: "Unit") -> bool:
    if unit is None:
        return False
    return int(getattr(unit, "cargo_max", 0) or 0) > 0 and int(
        getattr(unit, "cargo_used", 0) or 0
    ) > 0

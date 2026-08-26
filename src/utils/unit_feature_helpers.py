"""Shared unit-level feature helpers used by observation builders.

Extracted from ObservationTransformer so squad and other builders can call them
without importing the full transformer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, FrozenSet, List, Optional, Tuple

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

# Corruptors attack air with the normal weapon; buildings in the same cell use
# Caustic Spray (no anti-ground weapon).
_CAUSTIC_SPRAY_UNIT_TYPES: FrozenSet = (
    frozenset({UnitTypeId.CORRUPTOR}) if _SC2_AVAILABLE else frozenset()
)

# Cyclone Lock On is a targeted ability. Ladder units spawn with autocast ON,
# so the engine picks a target independently of the policy click. Attack
# hard-target maps to Lock On; autocast is turned off after spawn.
_LOCK_ON_UNIT_TYPES: FrozenSet = (
    frozenset({UnitTypeId.CYCLONE}) if _SC2_AVAILABLE else frozenset()
)
_LOCK_ON_ABILITY = AbilityId.LOCKON_LOCKON if _SC2_AVAILABLE else None

# Policy ``aoe_local``: point-targeted splash casts (agent-faithful Point2).
_AOE_POINT_ABILITIES: dict = (
    {
        UnitTypeId.INFESTOR: AbilityId.FUNGALGROWTH_FUNGALGROWTH,
        UnitTypeId.INFESTORBURROWED: AbilityId.FUNGALGROWTH_FUNGALGROWTH,
        UnitTypeId.HIGHTEMPLAR: AbilityId.PSISTORM_PSISTORM,
        UnitTypeId.GHOST: AbilityId.EMP_EMP,
        UnitTypeId.RAVAGER: AbilityId.EFFECT_CORROSIVEBILE,
    }
    if _SC2_AVAILABLE
    else {}
)

# Parasitic Bomb is unit-targeted (air in the clicked cell).
_AOE_AIR_UNIT_ABILITIES: dict = (
    {UnitTypeId.VIPER: AbilityId.PARASITICBOMB_PARASITICBOMB}
    if _SC2_AVAILABLE
    else {}
)

# Energy needed for ``aoe_local`` (Ravager bile is cooldown-only).
_AOE_ENERGY_COST: dict = (
    {
        UnitTypeId.INFESTOR: 75.0,
        UnitTypeId.INFESTORBURROWED: 75.0,
        UnitTypeId.HIGHTEMPLAR: 75.0,
        UnitTypeId.GHOST: 75.0,
        UnitTypeId.VIPER: 125.0,
        UnitTypeId.RAVAGER: 0.0,
    }
    if _SC2_AVAILABLE
    else {}
)

# Controllable Purification Nova ball (steer with move / attack→move).
_PURIFICATION_NOVA_BALL_TYPES: FrozenSet = (
    frozenset({UnitTypeId.DISRUPTORPHASED}) if _SC2_AVAILABLE else frozenset()
)

# Carrier Interceptors are auto-managed (launch / recall / rebuild). They must
# not enter squad pointer slots or player unit lists: a few Carriers spawn
# enough Interceptors to fill ``MAX_SQUAD_UNITS`` and crowd out real army tags.
_INTERCEPTOR_UNIT_TYPES: FrozenSet = (
    frozenset({UnitTypeId.INTERCEPTOR}) if _SC2_AVAILABLE else frozenset()
)

# Mid-episode controllable summons → preferred same-owner parent types.
# Interceptors are intentionally omitted (Carrier AI owns them). Structures
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
    """True for timed summons excluded from combat HP / kill reward accounting.

    Own side: natural expiration must not look like attrition.
    Enemy side: launch/recall/despawn (especially Carrier Interceptors) must not
    look like damage_reward / kill_reward.
    """
    return getattr(unit, "type_id", None) in _TEMPORARY_ATTRITION_UNIT_TYPES


def is_carrier_interceptor(unit: "Unit") -> bool:
    """True for Carrier Interceptors (auto-managed; never squad-selectable)."""
    return getattr(unit, "type_id", None) in _INTERCEPTOR_UNIT_TYPES


def counts_for_combat_hp_reward(unit: "Unit") -> bool:
    """Whether *unit*'s HP belongs in combat damage/kill reward baselines.

    Structures and timed summons (Interceptors, locusts, …) are excluded so
    launch/recall/despawn is not treated as army damage. Damage *dealt by*
    Interceptors still counts: it reduces permanent units' HP, and those
    units remain in the ledger.
    """
    if unit is None:
        return False
    # ``is True`` avoids MagicMock truthiness in unit tests.
    if getattr(unit, "is_structure", False) is True:
        return False
    return not is_temporary_attrition_unit(unit)


def uses_spawn_locusts_for_attack(unit: "Unit") -> bool:
    """True when policy attack should cast Spawn Locusts instead of ``unit.attack``."""
    return getattr(unit, "type_id", None) in _SPAWN_LOCUSTS_UNIT_TYPES


def uses_purification_nova_for_attack(unit: "Unit") -> bool:
    """True when policy attack should cast Purification Nova instead of ``unit.attack``."""
    return getattr(unit, "type_id", None) in _PURIFICATION_NOVA_UNIT_TYPES


def uses_caustic_spray_on_structures(unit: "Unit") -> bool:
    """True when policy attack on a building should cast Caustic Spray."""
    return getattr(unit, "type_id", None) in _CAUSTIC_SPRAY_UNIT_TYPES


def uses_lock_on_for_attack(unit: "Unit") -> bool:
    """True when policy attack on a unit should cast Cyclone Lock On."""
    return getattr(unit, "type_id", None) in _LOCK_ON_UNIT_TYPES


def _finite_unit_list(group: Any) -> List[Any]:
    """Iterate *group* at most ``len(group)`` times (guards MagicMock)."""
    if group is None:
        return []
    try:
        n = len(group)
    except TypeError:
        return []
    if n <= 0:
        return []
    out: List[Any] = []
    for i, unit in enumerate(group):
        if i >= n:
            break
        out.append(unit)
    return out


async def disable_cyclone_lock_on_autocast(bot: Any) -> int:
    """Turn Lock On autocast off for cyclones not yet processed.

    Ladder Cyclones spawn with Lock On autocast enabled, so the engine assigns
    a target without a policy hard-target. Toggle once per unit tag.

    Returns how many cyclones were toggled this call.
    """
    if not _SC2_AVAILABLE or bot is None or _LOCK_ON_ABILITY is None:
        return 0
    seen = getattr(bot, "_cyclone_lockon_autocast_off_tags", None)
    if not isinstance(seen, set):
        seen = set()
        bot._cyclone_lockon_autocast_off_tags = seen
    group = _finite_unit_list(getattr(bot, "all_units", None))
    if not group:
        group = _finite_unit_list(getattr(bot, "units", None)) + _finite_unit_list(
            getattr(bot, "enemy_units", None)
        )
    new_units: List[Any] = []
    for unit in group:
        if getattr(unit, "type_id", None) not in _LOCK_ON_UNIT_TYPES:
            continue
        tag = getattr(unit, "tag", None)
        if tag is None or tag in seen:
            continue
        new_units.append(unit)
        seen.add(tag)
    if not new_units:
        return 0
    client = getattr(bot, "client", None)
    toggle = getattr(client, "toggle_autocast", None) if client is not None else None
    if toggle is None:
        return 0
    result = toggle(new_units, _LOCK_ON_ABILITY)
    if hasattr(result, "__await__"):
        await result
    return len(new_units)


def uses_aoe_spell(unit: "Unit") -> bool:
    """True when policy ``aoe_local`` should cast a splash spell for this type."""
    tid = getattr(unit, "type_id", None)
    return tid in _AOE_POINT_ABILITIES or tid in _AOE_AIR_UNIT_ABILITIES


def unit_can_aoe(unit: "Unit", enemy_race: str | None = None) -> bool:
    """True when this unit can usefully issue ``aoe_local`` this step.

    Energy-gated for Infestor / High Templar / Ghost / Viper. Ravager bile has
    no energy cost. Ghost EMP is only flagged vs Protoss (shields / high energy).
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    if tid not in _AOE_ENERGY_COST:
        return False
    if tid == UnitTypeId.GHOST:
        if str(enemy_race or "").lower().strip() != "protoss":
            return False
    energy = float(getattr(unit, "energy", 0.0) or 0.0)
    return energy >= float(_AOE_ENERGY_COST[tid])


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

    Locusts, Broodlings, Nova balls, Adept shades, Infested Terrans, and
    Changelings are created mid-episode and are absent from spawn-time
    ``unit_pool_index``. Without adoption they never appear in squad pointer slots.
    Carrier Interceptors are not adopted (Carrier AI manages them; adopting them
    floods the 24-slot pointer). Combat HP rewards ignore timed summons via
    ``is_temporary_attrition_unit`` / ``counts_for_combat_hp_reward``.
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


def unit_center_in_spatial_cell(
    unit: Any,
    target: "Point2",
    cell_half: float = 0.5,
) -> bool:
    """True when *unit*'s center lies in the world cell around *target*.

    Same contract as CTF heal/load cell resolve (``cell_half`` game units).
    """
    pos = getattr(unit, "position", None)
    if pos is None or target is None:
        return False
    return (
        abs(float(pos.x) - float(target.x)) < cell_half
        and abs(float(pos.y) - float(target.y)) < cell_half
    )


# 2×2 buildings (Pylon, Supply Depot, …). Used when the unit has no catalog radius.
_DEFAULT_STRUCTURE_FOOTPRINT_HALF = 1.0


def _structure_footprint_half(structure: Any) -> float:
    """Half-width of a building's square footprint in game units."""
    for attr in ("footprint_radius", "radius"):
        value = getattr(structure, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return max(float(value), _DEFAULT_STRUCTURE_FOOTPRINT_HALF)
    return _DEFAULT_STRUCTURE_FOOTPRINT_HALF


def _structure_click_dist_sq(structure: Any, target: "Point2") -> Optional[float]:
    """Squared distance from *target* if the click lies on the building footprint."""
    if structure is None or target is None:
        return None
    if getattr(structure, "is_alive", True) is False:
        return None
    pos = getattr(structure, "position", None)
    if pos is None:
        return None
    half = _structure_footprint_half(structure)
    dx = float(pos.x) - float(target.x)
    dy = float(pos.y) - float(target.y)
    if abs(dx) >= half or abs(dy) >= half:
        return None
    return dx * dx + dy * dy


_GOD_MODE_UNIT_ATTRS = (
    "all_units",
    "units",
    "structures",
    "enemy_units",
    "enemy_structures",
)


def _player_owner_id(obj: Any) -> Optional[int]:
    """SC2 player id (1–15), or None when missing (typical unit-test mocks)."""
    for attr in ("owner_id", "owner"):
        value = getattr(obj, attr, None)
        if isinstance(value, int) and 1 <= value <= 15:
            return value
    return None


def _enemy_attack_candidates(bot: Any, caster: Any = None) -> List[Any]:
    """Hostiles for an attack click: caster-relative in God-mode CTF.

    Alliance buckets (``units`` vs ``enemy_units``) are not reliable after
    ``debug_control_enemy``: P2 may sit in ``enemy_units``, in ``units``
    (Self), or only in ``all_units`` (Ally). Filter by ``owner_id`` instead.
    """
    if caster is not None:
        return _hostile_candidates(bot, caster)
    return _candidates_from_attrs(bot, ("enemy_units", "enemy_structures"))


def _candidates_from_attrs(bot: Any, attrs: tuple) -> List[Any]:
    """Collect units from named bot collections (skips non-iterable mocks)."""
    out: List[Any] = []
    for attr in attrs:
        group = getattr(bot, attr, None)
        if group is None:
            continue
        if isinstance(group, (list, tuple, set)):
            out.extend(group)
            continue
        tags = getattr(group, "tags", None)
        if isinstance(tags, (set, frozenset)):
            try:
                out.extend(list(group))
            except TypeError:
                continue
    return out


def _dedupe_units(units: List[Any]) -> List[Any]:
    seen: set = set()
    out: List[Any] = []
    for unit in units:
        tag = getattr(unit, "tag", None)
        if tag is not None:
            if tag in seen:
                continue
            seen.add(tag)
        out.append(unit)
    return out


def _hostile_candidates(bot: Any, caster: Any) -> List[Any]:
    """Hostiles relative to *caster* owner, across God-mode alliance buckets."""
    caster_owner = _player_owner_id(caster)
    if caster_owner is None:
        return _candidates_from_attrs(bot, ("enemy_units", "enemy_structures"))
    return _dedupe_units(
        [
            unit
            for unit in _candidates_from_attrs(bot, _GOD_MODE_UNIT_ATTRS)
            if (other := _player_owner_id(unit)) is not None and other != caster_owner
        ]
    )


def allied_candidates_for_owner(bot: Any, owner_id: int) -> List[Any]:
    """Same-owner units/buildings across God-mode alliance buckets."""
    if not isinstance(owner_id, int):
        return []
    return _dedupe_units(
        [
            unit
            for unit in _candidates_from_attrs(bot, _GOD_MODE_UNIT_ATTRS)
            if _player_owner_id(unit) == owner_id
        ]
    )


def alliance_perspective_groups(bot: Any, p_id: int) -> Tuple[Any, Any, Any, Any]:
    """``(own_units, enemy_units, own_structs, enemy_structs)`` from alliance buckets."""
    player_id = int(getattr(bot, "player_id", 1) or 1)
    if int(p_id) == player_id:
        return (
            getattr(bot, "units", None),
            getattr(bot, "enemy_units", None),
            getattr(bot, "structures", None),
            getattr(bot, "enemy_structures", None),
        )
    return (
        getattr(bot, "enemy_units", None),
        getattr(bot, "units", None),
        getattr(bot, "enemy_structures", None),
        getattr(bot, "structures", None),
    )


def perspective_unit_groups(bot: Any, p_id: int) -> Tuple[Any, Any, Any, Any]:
    """Own/enemy mobiles and structures for *p_id*.

    After ``debug_control_enemy``, alliance buckets (``units`` vs
    ``enemy_units``) mix both armies. Prefer ``owner_id``. Fall back to
    alliance buckets when no unit carries a 1/2 owner tag — unit tests often
    omit owner ids.

    Use owner split whenever *either* side is tagged, including a wiped
    enemy army: otherwise a player-2 trainee falls back to the swapped
    buckets and paints its own leftover units as enemies.
    """
    own_id = int(p_id)
    enemy_id = 2 if own_id == 1 else 1
    own_all = allied_candidates_for_owner(bot, own_id)
    enemy_all = allied_candidates_for_owner(bot, enemy_id)
    if own_all or enemy_all:
        own_units = [u for u in own_all if not getattr(u, "is_structure", False)]
        enemy_units = [u for u in enemy_all if not getattr(u, "is_structure", False)]
        own_structs = [u for u in own_all if getattr(u, "is_structure", False)]
        enemy_structs = [u for u in enemy_all if getattr(u, "is_structure", False)]
        return own_units, enemy_units, own_structs, enemy_structs
    return alliance_perspective_groups(bot, p_id)


def first_unit(group: Any) -> Optional[Any]:
    """First member of a Units collection or list, or None."""
    if group is None:
        return None
    if isinstance(group, (list, tuple)):
        return group[0] if group else None
    if getattr(group, "exists", False):
        return getattr(group, "first", None)
    return None


def first_unit_of_type(group: Any, type_id: Any) -> Optional[Any]:
    """First unit matching *type_id* in a Units collection or list."""
    if group is None or type_id is None:
        return None
    if callable(group) and not isinstance(group, (list, tuple)):
        try:
            col = group(type_id)
            found = first_unit(col)
            if found is not None:
                return found
            if col is None:
                return None
            for unit in col:
                if getattr(unit, "type_id", None) == type_id:
                    return unit
        except Exception:
            return None
        return None
    try:
        for unit in group:
            if getattr(unit, "type_id", None) == type_id:
                return unit
    except TypeError:
        return None
    return None


def _issue_weapon_order(
    bot: Any,
    unit: Any,
    world_point: "Point2",
    hard_target: Any,
) -> Optional[Any]:
    """Hard-target a hostile by owner, else attack-move to *world_point*.

    God-mode orders go out on the P1 client, but empty-cell P2 must still
    attack-move: most spatial clicks miss a unit center, and ``move`` made
    P2 walk through fights (returns collapsed, explained variance ~-1).
    Same-owner units are never hard-targets.
    """
    target_owner = _player_owner_id(hard_target) if hard_target is not None else None
    caster_owner = _player_owner_id(unit)
    if (
        hard_target is not None
        and target_owner is not None
        and target_owner == caster_owner
    ):
        hard_target = None
    if hard_target is not None:
        bot.do(unit.attack(hard_target))
        return hard_target
    bot.do(unit.attack(world_point))
    return None


def _enemy_dist_sq_in_cell(
    enemy: Any,
    target: "Point2",
    cell_half: float,
) -> Optional[float]:
    """Squared distance from *target* if *enemy* is alive and in the cell."""
    if enemy is None:
        return None
    if getattr(enemy, "is_alive", True) is False:
        return None
    if not unit_center_in_spatial_cell(enemy, target, cell_half=cell_half):
        return None
    pos = enemy.position
    dx = float(pos.x) - float(target.x)
    dy = float(pos.y) - float(target.y)
    return dx * dx + dy * dy


def resolve_enemy_at_point(
    bot: Any,
    target: "Point2",
    cell_half: float = 0.5,
    caster: Any = None,
) -> Optional[Any]:
    """Enemy unit or building whose center lies in the clicked spatial cell.

    If several occupy the cell, pick the closest center to *target*.
    Returns ``None`` when the cell is empty (caller should attack-move / P2 move).
    ``caster`` selects the hostile list (owner-filtered God-mode armies).
    """
    best = None
    best_dist_sq = float("inf")
    for enemy in _enemy_attack_candidates(bot, caster):
        dist_sq = _enemy_dist_sq_in_cell(enemy, target, cell_half)
        if dist_sq is None:
            continue
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best = enemy
    return best


def resolve_corruptor_target_at_point(
    bot: Any,
    target: "Point2",
    cell_half: float = 0.5,
    caster: Any = None,
) -> Optional[Any]:
    """Corruptor hard-target: air in the clicked cell, else a building under the click.

    Flying non-structures always beat structures even when the building is
    closer to the click. Ground units are not valid Corruptor targets.

    Buildings use footprint overlap (Pylon is 2×2), not the 1×1 unit-center
    cell. ``unit.attack(Point2)`` does not Caustic Spray, so a center-only
    match left most pylon clicks as a no-op air attack-move.
    """
    best_air = None
    best_air_dist_sq = float("inf")
    best_structure = None
    best_structure_dist_sq = float("inf")
    for enemy in _enemy_attack_candidates(bot, caster):
        is_structure = bool(getattr(enemy, "is_structure", False))
        is_flying = bool(getattr(enemy, "is_flying", False))
        if is_flying and not is_structure:
            dist_sq = _enemy_dist_sq_in_cell(enemy, target, cell_half)
            if dist_sq is None:
                continue
            if dist_sq < best_air_dist_sq:
                best_air_dist_sq = dist_sq
                best_air = enemy
        elif is_structure:
            dist_sq = _structure_click_dist_sq(enemy, target)
            if dist_sq is None:
                continue
            if dist_sq < best_structure_dist_sq:
                best_structure_dist_sq = dist_sq
                best_structure = enemy
    return best_air if best_air is not None else best_structure


def issue_attack_at_point(bot: Any, unit: "Unit", target: "Point2") -> Optional[Any]:
    """Issue an offensive order at *target*.

    Swarm Hosts (and burrowed) cast ``EFFECT_SPAWNLOCUSTS`` at the same point;
    Disruptors cast ``EFFECT_PURIFICATIONNOVA`` at the same point;
    Purification Nova balls ``move`` to the point (no weapon);
    Corruptors hard-target air in the cell with ``unit.attack``, or Caustic Spray
    a building whose footprint contains the click (Pylon is 2×2; a 1×1
    center-cell match missed most building clicks);
    Cyclones Lock On a hard-targeted enemy (autocast is disabled separately);
    all other units hard-target an enemy whose center is in the clicked cell
    via ``unit.attack(enemy)``, else attack-move with ``unit.attack(Point2)``.
    Same-owner units are never hard-targets (God-mode alliance buckets).

    Returns the hard-targeted enemy when one was issued; otherwise ``None``.
    """
    if uses_spawn_locusts_for_attack(unit):
        bot.do(unit(AbilityId.EFFECT_SPAWNLOCUSTS, target))
        return None
    if uses_purification_nova_for_attack(unit):
        bot.do(unit(AbilityId.EFFECT_PURIFICATIONNOVA, target))
        return None
    if is_purification_nova_ball(unit):
        bot.do(unit.move(target))
        return None
    if uses_caustic_spray_on_structures(unit):
        enemy = resolve_corruptor_target_at_point(bot, target, caster=unit)
        if enemy is not None and getattr(enemy, "is_structure", False):
            bot.do(unit(AbilityId.CAUSTICSPRAY_CAUSTICSPRAY, enemy))
            return enemy
        return _issue_weapon_order(bot, unit, target, enemy)
    enemy = resolve_enemy_at_point(bot, target, caster=unit)
    if uses_lock_on_for_attack(unit) and enemy is not None:
        bot.do(unit(AbilityId.LOCKON_LOCKON, enemy))
        return enemy
    return _issue_weapon_order(bot, unit, target, enemy)


def resolve_air_hostile_at_point(
    bot: Any,
    caster: "Unit",
    target: "Point2",
    cell_half: float = 0.5,
) -> Optional[Any]:
    """Flying hostile (non-structure) whose center lies in the clicked cell."""
    best = None
    best_dist_sq = float("inf")
    for enemy in _hostile_candidates(bot, caster):
        if bool(getattr(enemy, "is_structure", False)):
            continue
        if not bool(getattr(enemy, "is_flying", False)):
            continue
        dist_sq = _enemy_dist_sq_in_cell(enemy, target, cell_half)
        if dist_sq is None:
            continue
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best = enemy
    return best


def issue_aoe_at_point(bot: Any, unit: "Unit", target: "Point2") -> Optional[Any]:
    """Issue the caster's splash spell at *target* (``aoe_local``).

    Infestor / burrowed Infestor → Fungal Growth at the point;
    High Templar → Psi Storm at the point;
    Ghost → EMP at the point;
    Ravager → Corrosive Bile at the point;
    Viper → Parasitic Bomb on an air unit in the clicked cell (no-op if none);
    all other units no-op (do not convert to move).

    Returns the bomb hard-target when Parasitic Bomb was issued; otherwise ``None``.
    """
    tid = getattr(unit, "type_id", None)
    air_ability = _AOE_AIR_UNIT_ABILITIES.get(tid)
    if air_ability is not None:
        enemy = resolve_air_hostile_at_point(bot, unit, target)
        if enemy is None:
            return None
        bot.do(unit(air_ability, enemy))
        return enemy
    point_ability = _AOE_POINT_ABILITIES.get(tid)
    if point_ability is not None:
        bot.do(unit(point_ability, target))
    return None


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

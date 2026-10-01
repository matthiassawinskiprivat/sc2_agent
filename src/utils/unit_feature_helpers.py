"""Shared unit-level feature helpers used by observation builders.

Extracted from ObservationTransformer so squad and other builders can call them
without importing the full transformer.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, FrozenSet, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from sc2.position import Point2
    from sc2.unit import Unit


try:
    from sc2.ids.ability_id import AbilityId
    from sc2.ids.buff_id import BuffId
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

# heal_local click box. Omit ``env.heal_click_cells`` → 1 cell (±0.5), same as
# load/attack. Set it only when the curriculum uses a coarser square.
HEAL_CELL_HALF = 0.5
DEFAULT_HEAL_CLICK_CELLS = 1.0
MIN_HEAL_CLICK_CELLS = 1.0
MAX_HEAL_CLICK_CELLS = 5.0


def heal_click_cells_from_conf(training_conf: dict | None) -> float:
    """Heal click square width in spatial cells. Default 1 (1×1 pointer)."""
    conf = training_conf or {}
    env = conf.get("env") if isinstance(conf.get("env"), dict) else {}
    raw = conf.get("heal_click_cells", env.get("heal_click_cells", DEFAULT_HEAL_CLICK_CELLS))
    try:
        cells = float(raw)
    except (TypeError, ValueError):
        cells = DEFAULT_HEAL_CLICK_CELLS
    if cells != cells:  # NaN
        cells = DEFAULT_HEAL_CLICK_CELLS
    return min(max(cells, MIN_HEAL_CLICK_CELLS), MAX_HEAL_CLICK_CELLS)


def heal_click_cell_half(training_conf: dict | None) -> float:
    """Half-width of the heal click square in game units."""
    return heal_click_cells_from_conf(training_conf) / 2.0

# Same radii as ObservationTransformer._draw_unit_healing (HPS field).
# Ability radii for the HPS observation field (not an actuation range gate).
HEAL_CAST_RANGE: dict = (
    {
        UnitTypeId.MEDIVAC: 4.0,
        UnitTypeId.QUEEN: 7.0,
        UnitTypeId.SCV: 1.0,
    }
    if _SC2_AVAILABLE
    else {}
)

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

# AoE spells that require research before the ability exists.
_AOE_REQUIRED_UPGRADE: dict = (
    {UnitTypeId.HIGHTEMPLAR: UpgradeId.PSISTORMTECH} if _SC2_AVAILABLE else {}
)

# Ability id used for cooldown / availability checks (same as actuation).
_AOE_ABILITY_BY_TYPE: dict = (
    {
        **_AOE_POINT_ABILITIES,
        **_AOE_AIR_UNIT_ABILITIES,
    }
    if _SC2_AVAILABLE
    else {}
)

# Policy ``snipe``: unit-targeted specials (enemy pointer + per-type ability).
# Ghost Steady Targeting, BC Yamato, Infestor Neural, HT Feedback.
_SNIPE_ABILITIES: dict = (
    {
        UnitTypeId.GHOST: AbilityId.EFFECT_GHOSTSNIPE,
        UnitTypeId.BATTLECRUISER: AbilityId.YAMATO_YAMATOGUN,
        UnitTypeId.INFESTOR: AbilityId.NEURALPARASITE_NEURALPARASITE,
        UnitTypeId.INFESTORBURROWED: AbilityId.NEURALPARASITE_NEURALPARASITE,
        UnitTypeId.HIGHTEMPLAR: AbilityId.FEEDBACK_FEEDBACK,
    }
    if _SC2_AVAILABLE
    else {}
)

# Energy gate for ``snipe`` (Yamato is cooldown-only).
_SNIPE_ENERGY_COST: dict = (
    {
        UnitTypeId.GHOST: 50.0,
        UnitTypeId.BATTLECRUISER: 0.0,
        UnitTypeId.INFESTOR: 100.0,
        UnitTypeId.INFESTORBURROWED: 100.0,
        UnitTypeId.HIGHTEMPLAR: 50.0,
    }
    if _SC2_AVAILABLE
    else {}
)

# Snipe spells that require research before the ability exists.
_SNIPE_REQUIRED_UPGRADE: dict = (
    {
        UnitTypeId.INFESTOR: UpgradeId.NEURALPARASITE,
        UnitTypeId.INFESTORBURROWED: UpgradeId.NEURALPARASITE,
        UnitTypeId.BATTLECRUISER: UpgradeId.YAMATOCANNON,
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


async def _disable_autocast_for_unit_types(
    bot: Any,
    unit_types: FrozenSet,
    ability: Any,
    seen_attr: str,
) -> int:
    """Toggle *ability* autocast off once per matching unit tag."""
    if not _SC2_AVAILABLE or bot is None or ability is None:
        return 0
    seen = getattr(bot, seen_attr, None)
    if not isinstance(seen, set):
        seen = set()
        setattr(bot, seen_attr, seen)
    group = _finite_unit_list(getattr(bot, "all_units", None))
    if not group:
        group = _finite_unit_list(getattr(bot, "units", None)) + _finite_unit_list(
            getattr(bot, "enemy_units", None)
        )
    new_units: List[Any] = []
    for unit in group:
        if getattr(unit, "type_id", None) not in unit_types:
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
    result = toggle(new_units, ability)
    if hasattr(result, "__await__"):
        await result
    return len(new_units)


async def disable_cyclone_lock_on_autocast(bot: Any) -> int:
    """Turn Lock On autocast off for cyclones not yet processed.

    Ladder Cyclones spawn with Lock On autocast enabled, so the engine assigns
    a target without a policy hard-target. Toggle once per unit tag.

    Returns how many cyclones were toggled this call.
    """
    return await _disable_autocast_for_unit_types(
        bot,
        _LOCK_ON_UNIT_TYPES,
        _LOCK_ON_ABILITY,
        "_cyclone_lockon_autocast_off_tags",
    )


def uses_aoe_spell(unit: "Unit") -> bool:
    """True when policy ``aoe_local`` should cast a splash spell for this type."""
    tid = getattr(unit, "type_id", None)
    return tid in _AOE_POINT_ABILITIES or tid in _AOE_AIR_UNIT_ABILITIES


# Control buff on the stolen body. The caster carries NEURALPARASITECHILDREN
# while the link is up, and NEURALPARASITEWAIT is only the cast windup.
_NEURAL_CONTROL_BUFF = BuffId.NEURALPARASITE if _SC2_AVAILABLE else None


def unit_is_neuraled(unit: "Unit") -> bool:
    """True when *unit* is the body under Infestor Neural Parasite control.

    Own army units stay 0, including the Infestor that cast it. The cast-windup
    buff and the caster's child buff are not this flag.
    """
    if unit is None or _NEURAL_CONTROL_BUFF is None:
        return False
    buffs = getattr(unit, "buffs", None)
    # python-sc2 exposes a set of BuffId. MagicMock units have no real buff list.
    if not isinstance(buffs, (list, tuple, set, frozenset)):
        return False
    control = _NEURAL_CONTROL_BUFF
    control_value = getattr(control, "value", None)
    for buff in buffs:
        if buff == control:
            return True
        if (
            isinstance(buff, int)
            and not isinstance(buff, bool)
            and buff == control_value
        ):
            return True
        value = getattr(buff, "value", None)
        if isinstance(value, int) and not isinstance(value, bool) and value == control_value:
            return True
    return False


def neural_spell_role(unit: "Unit", perspective_player_id: Optional[int]) -> str:
    """Who may cast with a Neural Parasite body from this squad's point of view.

    ``own`` — not neuralled, or the caller did not pass a perspective.
    ``borrowed`` — live owner is this squad, so it may cast the stolen spells.
    ``stolen_away`` — the body still sits in the original squad, which must not cast.
    """
    if perspective_player_id is None or not unit_is_neuraled(unit):
        return "own"
    try:
        owner = int(getattr(unit, "owner_id", 0) or 0)
    except (TypeError, ValueError):
        return "own"
    if owner == int(perspective_player_id):
        return "borrowed"
    return "stolen_away"


def neural_command_allowed(
    unit: "Unit", perspective_player_id: Optional[int], head_name: str
) -> bool:
    """Spell orders follow the live neural owner; other orders do not.

    The controller may issue ``aoe_local`` / ``snipe`` on the stolen caster and
    nothing else. The original squad keeps move and attack, but not those spells.
    """
    role = neural_spell_role(unit, perspective_player_id)
    if role == "own":
        return True
    is_spell = "aoe" in str(head_name) or "snipe" in str(head_name)
    if role == "borrowed":
        return is_spell
    return not is_spell


def unit_is_squad_spellcaster(unit: "Unit") -> bool:
    """True when this type has an ``aoe_local`` or ``snipe`` ability."""
    tid = getattr(unit, "type_id", None)
    return tid in _AOE_ENERGY_COST or tid in _SNIPE_ENERGY_COST


def borrowed_neural_spellcaster_tags(
    units: Sequence[Any],
    perspective_player_id: int,
    native_tags: Sequence[int],
) -> List[int]:
    """Tags of spellcasters this player controls via Neural Parasite.

    Native pool tags are excluded. Non-casters (a neuralled Zealot) are excluded
    so they do not take a pointer slot the policy cannot cast with.
    """
    native = {int(tag) for tag in native_tags}
    found: List[int] = []
    for unit in units:
        if unit is None or not unit_is_neuraled(unit):
            continue
        if getattr(unit, "is_structure", False):
            continue
        if not unit_is_squad_spellcaster(unit):
            continue
        if neural_spell_role(unit, perspective_player_id) != "borrowed":
            continue
        tag = getattr(unit, "tag", None)
        if tag is None:
            continue
        tag_i = int(tag)
        if tag_i in native or tag_i in found:
            continue
        found.append(tag_i)
    found.sort()
    return found


def squad_pointer_tags(
    native_tags: Sequence[int],
    borrowed_tags: Sequence[int],
    *,
    limit: Optional[int] = None,
) -> List[int]:
    """Native squad tags first, then borrowed spellcasters in the leftover slots.

    Appending keeps existing slot indices stable when a neural link starts.
    A full squad does not drop a native unit to make room.
    """
    if limit is None:
        from src.agent.squad_constants import MAX_SQUAD_UNITS

        limit = MAX_SQUAD_UNITS
    native: List[int] = []
    seen = set()
    for tag in sorted(int(t) for t in native_tags):
        if tag in seen:
            continue
        seen.add(tag)
        native.append(tag)
        if len(native) >= limit:
            return native
    extra: List[int] = []
    for tag in sorted(int(t) for t in borrowed_tags):
        if tag in seen:
            continue
        seen.add(tag)
        extra.append(tag)
    room = limit - len(native)
    return native + extra[:room]


def unit_neural_remaining(unit: "Unit") -> float:
    """Fraction of Neural Parasite still left, in ``[0, 1]``.

    The client exposes one visible timer bar, ``buff_duration_remain`` over
    ``buff_duration_max``, in game steps. It is not a duration per buff. The
    ratio is read only while the control buff is on the body, so a different
    timer (stim, a shield) cannot fill this channel. A fresh cast is 1; the
    bar reaching 0 is 0. If the buff is present but the client reports no
    timer, the value stays 1 until the buff drops.
    """
    if not unit_is_neuraled(unit):
        return 0.0
    remain = getattr(unit, "buff_duration_remain", None)
    maximum = getattr(unit, "buff_duration_max", None)
    if (
        isinstance(remain, bool)
        or isinstance(maximum, bool)
        or not isinstance(remain, (int, float))
        or not isinstance(maximum, (int, float))
    ):
        return 1.0
    if float(maximum) <= 0.0:
        return 1.0
    return min(1.0, max(0.0, float(remain) / float(maximum)))


def uses_snipe_spell(unit: "Unit") -> bool:
    """True when policy ``snipe`` should cast a unit-targeted special for this type."""
    tid = getattr(unit, "type_id", None)
    return tid in _SNIPE_ABILITIES


def unit_can_snipe(
    unit: "Unit",
    upgrades: frozenset | None = None,
    bot: Any = None,
) -> bool:
    """True when this unit can usefully issue ``snipe`` this step.

    Energy-gated for Ghost / Infestor / High Templar. Battlecruiser Yamato is
    cooldown-only but requires ``YAMATOCANNON`` research. Infestor Neural
    Parasite requires ``NEURALPARASITE``. When ``bot._available_abilities_by_tag``
    is present (opt-in ``query_spell_abilities``), the spell ability must appear
    there (CD only — energy is checked above). CTF skips that RPC by default.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    if tid not in _SNIPE_ENERGY_COST:
        return False
    required = _SNIPE_REQUIRED_UPGRADE.get(tid)
    if required is not None:
        if required not in (upgrades or frozenset()):
            return False
    energy = float(getattr(unit, "energy", 0.0) or 0.0)
    if energy < float(_SNIPE_ENERGY_COST[tid]):
        return False
    ability = _SNIPE_ABILITIES.get(tid)
    if ability is not None:
        abilities = _available_abilities_for_unit(unit, bot)
        if abilities is not None:
            return ability in abilities
        if _spell_ability_on_cooldown(unit, bot, "snipe"):
            return False
        if _unit_is_using_any_ability(unit, frozenset({ability})):
            return False
    return True


def snipe_target_is_eligible(caster: "Unit", target: "Unit") -> bool:
    """Per-ability legality for a ``snipe`` enemy-pointer target.

    Ghost Steady Targeting needs a biological non-structure; Feedback needs an
    energy unit; Neural / Yamato accept any hostile mobile (structures stay off
    the enemy pointer). Ownership is checked by the caller.
    """
    if caster is None or target is None:
        return False
    if bool(getattr(target, "is_structure", False)):
        return False
    tid = getattr(caster, "type_id", None)
    if tid == UnitTypeId.GHOST:
        return bool(getattr(target, "is_biological", False))
    if tid == UnitTypeId.HIGHTEMPLAR:
        return float(getattr(target, "energy_max", 0.0) or 0.0) > 0.0
    if tid in (
        UnitTypeId.BATTLECRUISER,
        UnitTypeId.INFESTOR,
        UnitTypeId.INFESTORBURROWED,
    ):
        return True
    return False


def snipe_same_owner_blocked(caster: "Unit", target: "Unit") -> bool:
    """True when ``snipe`` must not hit this target because it is a real friendly.

    Same owner is refused for the agent's own army. A different body under
    Neural Parasite stays legal, so two stolen Battlecruisers can Yamato each
    other on later steps. A caster never snipes its own tag.
    """
    if caster is None or target is None:
        return True
    try:
        same_owner = int(getattr(target, "owner_id", 0) or 0) == int(
            getattr(caster, "owner_id", 0) or 0
        )
    except (TypeError, ValueError):
        return False
    if not same_owner:
        return False
    caster_tag = getattr(caster, "tag", None)
    target_tag = getattr(target, "tag", None)
    if (
        caster_tag is not None
        and target_tag is not None
        and int(caster_tag) == int(target_tag)
    ):
        return True
    return not unit_is_neuraled(target)


def issue_snipe_on_unit(
    bot: Any, unit: "Unit", target: "Unit", *, stamp_cooldown: bool = True
) -> bool:
    """Issue the caster's ``snipe`` ability on *target*. Returns True if issued.

    ``stamp_cooldown`` records a local ready-at loop (Yamato / Snipe CD). Pass
    ``False`` on reissue so the timer is not reset every physics step.
    """
    if bot is None or unit is None or target is None:
        return False
    if snipe_same_owner_blocked(unit, target):
        return False
    if not snipe_target_is_eligible(unit, target):
        return False
    tid = getattr(unit, "type_id", None)
    ability = _SNIPE_ABILITIES.get(tid)
    if ability is None:
        return False
    bot.do(unit(ability, target))
    if stamp_cooldown:
        mark_spell_ability_cooldown(bot, unit, "snipe")
    return True


def unit_can_aoe(
    unit: "Unit",
    enemy_race: str | None = None,
    upgrades: frozenset | None = None,
    bot: Any = None,
    *,
    has_tactical_air_enemy: bool = True,
) -> bool:
    """True when this unit can usefully issue ``aoe_local`` this step.

    Energy-gated for Infestor / High Templar / Ghost / Viper. Ravager bile has
    no energy cost but still needs the ability off cooldown. Ghost EMP is only
    flagged vs Protoss. High Templar Psi Storm requires ``PSISTORMTECH``.
    Viper Parasitic Bomb needs a flying hostile within engage fovea of any ally
    (``has_tactical_air_enemy``). When ``bot._available_abilities_by_tag`` is
    present (opt-in ``query_spell_abilities``), the spell ability must appear
    there (CD only for AoE casters — energy is checked above). CTF skips that
    RPC by default.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    if tid not in _AOE_ENERGY_COST:
        return False
    required = _AOE_REQUIRED_UPGRADE.get(tid)
    if required is not None:
        if required not in (upgrades or frozenset()):
            return False
    if tid == UnitTypeId.GHOST:
        if str(enemy_race or "").lower().strip() != "protoss":
            return False
    if tid == UnitTypeId.VIPER and not has_tactical_air_enemy:
        return False
    energy = float(getattr(unit, "energy", 0.0) or 0.0)
    if energy < float(_AOE_ENERGY_COST[tid]):
        return False
    ability = _AOE_ABILITY_BY_TYPE.get(tid)
    if ability is not None:
        abilities = _available_abilities_for_unit(unit, bot)
        if abilities is not None:
            return ability in abilities
        if _spell_ability_on_cooldown(unit, bot, "aoe"):
            return False
        if _unit_is_using_any_ability(unit, frozenset({ability})):
            return False
    return True


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

    Same contract as CTF load/attack cell resolve (``cell_half`` game units).
    """
    pos = getattr(unit, "position", None)
    if pos is None or target is None:
        return False
    return (
        abs(float(pos.x) - float(target.x)) < cell_half
        and abs(float(pos.y) - float(target.y)) < cell_half
    )


def unit_click_radius(unit: Any) -> float:
    """Collision / selection radius for click-picking (0 if unknown)."""
    try:
        radius = float(getattr(unit, "radius", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return radius if radius > 0.0 else 0.0


def heal_click_hits_unit(
    unit: Any,
    target: "Point2",
    cell_half: float = HEAL_CELL_HALF,
) -> bool:
    """True when the heal click is on the unit body or its center is in the cell.

    Matches the client idea that a Thor is easier to click than a Cyclone.
    ``cell_half`` is half the configured square (1 cell → 0.5). Attack / load
    still use ``unit_center_in_spatial_cell`` at one cell only.
    """
    if unit_center_in_spatial_cell(unit, target, cell_half=cell_half):
        return True
    radius = unit_click_radius(unit)
    if radius <= 0.0 or target is None:
        return False
    pos = getattr(unit, "position", None)
    if pos is None:
        return False
    try:
        return float(pos.distance_to(target)) <= radius
    except (TypeError, ValueError, AttributeError):
        return False


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
    caster_tag = getattr(caster, "tag", None)
    if caster_owner is None:
        return _candidates_from_attrs(bot, ("enemy_units", "enemy_structures"))
    return _dedupe_units(
        [
            unit
            for unit in _candidates_from_attrs(bot, _GOD_MODE_UNIT_ATTRS)
            if _counts_as_attack_hostile(unit, caster_owner, caster_tag)
        ]
    )


def _counts_as_attack_hostile(unit: Any, caster_owner: int, caster_tag: Any) -> bool:
    """True when *unit* is a legal hard-target for this caster.

    Different ``owner_id`` is the normal hostile test. A body under Neural
    Parasite is hostile to both players, including the controller, so either
    side can choose to kill it. The caster is never its own target.
    """
    tag = getattr(unit, "tag", None)
    if caster_tag is not None and tag is not None and int(tag) == int(caster_tag):
        return False
    if unit_is_neuraled(unit):
        return True
    other = _player_owner_id(unit)
    return other is not None and other != caster_owner


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
        and not unit_is_neuraled(hard_target)
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


def issue_aoe_at_point(
    bot: Any, unit: "Unit", target: "Point2", *, stamp_cooldown: bool = True
) -> Optional[Any]:
    """Issue the caster's splash spell at *target* (``aoe_local``).

    Infestor / burrowed Infestor → Fungal Growth at the point;
    High Templar → Psi Storm at the point;
    Ghost → EMP at the point;
    Ravager → Corrosive Bile at the point;
    Viper → Parasitic Bomb on an air unit in the clicked cell (no-op if none);
    all other units no-op (do not convert to move).

    Returns the bomb hard-target when Parasitic Bomb was issued; otherwise ``None``.

    ``stamp_cooldown`` records a local ready-at loop (bile / storm / bomb CD).
    Pass ``False`` on leftover-order reissue so the timer is not reset.
    """
    tid = getattr(unit, "type_id", None)
    air_ability = _AOE_AIR_UNIT_ABILITIES.get(tid)
    if air_ability is not None:
        enemy = resolve_air_hostile_at_point(bot, unit, target)
        if enemy is None:
            return None
        bot.do(unit(air_ability, enemy))
        if stamp_cooldown:
            mark_spell_ability_cooldown(bot, unit, "aoe")
        return enemy
    point_ability = _AOE_POINT_ABILITIES.get(tid)
    if point_ability is not None:
        bot.do(unit(point_ability, target))
        if stamp_cooldown:
            mark_spell_ability_cooldown(bot, unit, "aoe")
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

# Surface types that can burrow (aligned with CaptureTheFlagBot._BURROW_DOWN_BY_TYPE).
_BURROW_DOWN_TYPES: FrozenSet = (
    frozenset(
        {
            UnitTypeId.ZERGLING,
            UnitTypeId.BANELING,
            UnitTypeId.ROACH,
            UnitTypeId.RAVAGER,
            UnitTypeId.HYDRALISK,
            UnitTypeId.INFESTOR,
            UnitTypeId.SWARMHOSTMP,
            UnitTypeId.ULTRALISK,
            UnitTypeId.QUEEN,
            UnitTypeId.DRONE,
            UnitTypeId.WIDOWMINE,
            UnitTypeId.LURKERMP,
        }
    )
    if _SC2_AVAILABLE
    else frozenset()
)

# Burrowed morphs that can unburrow (aligned with CaptureTheFlagBot._BURROW_UP_BY_TYPE).
_BURROW_UP_TYPES: FrozenSet = (
    frozenset(
        {
            UnitTypeId.ZERGLINGBURROWED,
            UnitTypeId.BANELINGBURROWED,
            UnitTypeId.ROACHBURROWED,
            UnitTypeId.RAVAGERBURROWED,
            UnitTypeId.HYDRALISKBURROWED,
            UnitTypeId.INFESTORBURROWED,
            UnitTypeId.SWARMHOSTBURROWEDMP,
            UnitTypeId.ULTRALISKBURROWED,
            UnitTypeId.QUEENBURROWED,
            UnitTypeId.DRONEBURROWED,
            UnitTypeId.WIDOWMINEBURROWED,
            UnitTypeId.LURKERMPBURROWED,
            UnitTypeId.LURKERBURROWED,
        }
    )
    if _SC2_AVAILABLE
    else frozenset()
)

# Unit types that burrow without the general BURROW tech (inherent ability).
_BURROW_WITHOUT_TECH: FrozenSet = (
    frozenset({UnitTypeId.WIDOWMINE, UnitTypeId.LURKERMP})
    if _SC2_AVAILABLE
    else frozenset()
)

# Fire from a planted / burrowed morph (lurker-style). Still receive attack orders
# even though ``unit_is_immobile`` is true.
_ATTACKS_WHILE_IMMOBILE: FrozenSet = (
    frozenset(
        {
            UnitTypeId.LURKERMPBURROWED,
            UnitTypeId.LURKERBURROWED,
            UnitTypeId.SIEGETANKSIEGED,
            UnitTypeId.LIBERATORAG,
            UnitTypeId.WIDOWMINEBURROWED,
        }
    )
    if _SC2_AVAILABLE
    else frozenset()
)

# Surface morphs that only attack after burrowing.
_ATTACK_ONLY_WHEN_BURROWED: FrozenSet = (
    frozenset({UnitTypeId.LURKERMP, UnitTypeId.WIDOWMINE})
    if _SC2_AVAILABLE
    else frozenset()
)

# Personal cloak types that require research.
_CLOAK_REQUIRED_UPGRADE: dict = (
    {
        UnitTypeId.GHOST: UpgradeId.PERSONALCLOAKING,
        UnitTypeId.BANSHEE: UpgradeId.BANSHEECLOAK,
    }
    if _SC2_AVAILABLE
    else {}
)

# Heal / repair readiness (``can_heal`` / ``can_repair`` scalars).
GAME_LOOPS_PER_SECOND = 22.4
HEAL_ABILITY_COOLDOWN_SECONDS: dict = (
    {
        UnitTypeId.QUEEN: 1.0,
        UnitTypeId.MEDIVAC: 1.0,
        UnitTypeId.SHIELDBATTERY: 1.0,
    }
    if _SC2_AVAILABLE
    else {}
)
# LotV Faster seconds. Energy-only spells get a 1s lockout so the mask does
# not reopen before the next observation shows the energy drop.
AOE_ABILITY_COOLDOWN_SECONDS: dict = (
    {
        UnitTypeId.RAVAGER: 7.14,
        UnitTypeId.HIGHTEMPLAR: 1.79,
        UnitTypeId.VIPER: 7.14,
        UnitTypeId.INFESTOR: 1.0,
        UnitTypeId.INFESTORBURROWED: 1.0,
        UnitTypeId.GHOST: 1.0,
    }
    if _SC2_AVAILABLE
    else {}
)
SNIPE_ABILITY_COOLDOWN_SECONDS: dict = (
    {
        UnitTypeId.GHOST: 14.3,
        UnitTypeId.BATTLECRUISER: 71.4,
        UnitTypeId.INFESTOR: 1.0,
        UnitTypeId.INFESTORBURROWED: 1.0,
        UnitTypeId.HIGHTEMPLAR: 1.0,
    }
    if _SC2_AVAILABLE
    else {}
)
_SPELL_CD_SECONDS = {
    "heal": HEAL_ABILITY_COOLDOWN_SECONDS,
    "aoe": AOE_ABILITY_COOLDOWN_SECONDS,
    "snipe": SNIPE_ABILITY_COOLDOWN_SECONDS,
}
_SPELL_CD_ATTR = {
    "heal": "_heal_ability_ready_after_loop",
    "aoe": "_aoe_ability_ready_after_loop",
    "snipe": "_snipe_ability_ready_after_loop",
}
_HEAL_ENERGY_MIN: dict = (
    {
        UnitTypeId.MEDIVAC: 5.0,
        UnitTypeId.QUEEN: 50.0,
        UnitTypeId.SHIELDBATTERY: 3.0,
    }
    if _SC2_AVAILABLE
    else {}
)
_HEAL_ABILITIES_BY_TYPE: dict = (
    {
        UnitTypeId.QUEEN: frozenset({AbilityId.TRANSFUSION_TRANSFUSION}),
        UnitTypeId.MEDIVAC: frozenset({AbilityId.MEDIVACHEAL_HEAL}),
        UnitTypeId.SHIELDBATTERY: frozenset(
            a
            for a in (
                getattr(
                    AbilityId,
                    "SHIELDBATTERYRECHARGEEX5_SHIELDBATTERYRECHARGE",
                    None,
                ),
            )
            if a is not None
        ),
    }
    if _SC2_AVAILABLE
    else {}
)
_REPAIR_ABILITIES: FrozenSet = (
    frozenset({AbilityId.EFFECT_REPAIR_SCV, AbilityId.EFFECT_REPAIR})
    if _SC2_AVAILABLE
    else frozenset()
)
_HEAL_CHANNEL_ABILITIES: FrozenSet = (
    frozenset(
        a
        for a in (
            AbilityId.MEDIVACHEAL_HEAL,
            AbilityId.TRANSFUSION_TRANSFUSION,
            getattr(
                AbilityId,
                "SHIELDBATTERYRECHARGEEX5_SHIELDBATTERYRECHARGE",
                None,
            ),
        )
        if a is not None
    )
    if _SC2_AVAILABLE
    else frozenset()
)
_REPAIR_CHANNEL_ABILITIES: FrozenSet = (
    frozenset(
        {
            AbilityId.EFFECT_REPAIR_SCV,
            AbilityId.EFFECT_REPAIR,
            AbilityId.EFFECT_REPAIR_MULE,
        }
    )
    if _SC2_AVAILABLE
    else frozenset()
)
_SCV_REPAIR_MIN_MINERALS = 1


def _unit_game_loop(bot: Any, unit: "Unit") -> int:
    state = getattr(bot, "state", None) if bot is not None else None
    raw = getattr(state, "game_loop", None) if state is not None else None
    if raw is None:
        raw = getattr(unit, "game_loop", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _freeze_ability_map(raw_cache: dict) -> dict:
    """Convert query results to frozensets once (hot path reads these many times)."""
    frozen: dict = {}
    for tag, raw in raw_cache.items():
        if raw is None:
            frozen[tag] = frozenset()
        elif isinstance(raw, frozenset):
            frozen[tag] = raw
        else:
            try:
                frozen[tag] = frozenset(raw)
            except TypeError:
                frozen[tag] = frozenset()
    return frozen


def _available_abilities_for_unit(unit: "Unit", bot: Any) -> Optional[FrozenSet]:
    if bot is None or unit is None:
        return None
    tag = int(getattr(unit, "tag", 0) or 0)
    if tag <= 0:
        return None
    cache = getattr(bot, "_available_abilities_by_tag", None)
    if not isinstance(cache, dict) or tag not in cache:
        return None
    raw = cache.get(tag)
    if raw is None:
        return frozenset()
    if isinstance(raw, frozenset):
        return raw
    # Tests / callers may stash a plain set; freeze on read as a fallback.
    try:
        return frozenset(raw)
    except TypeError:
        return frozenset()


def _heal_energy_sufficient(unit: "Unit") -> bool:
    tid = getattr(unit, "type_id", None)
    if tid not in _HEAL_ENERGY_MIN:
        return False
    try:
        energy = float(getattr(unit, "energy", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    minimum = float(_HEAL_ENERGY_MIN[tid])
    if tid == UnitTypeId.MEDIVAC:
        return energy > minimum
    if tid == UnitTypeId.SHIELDBATTERY:
        return energy > minimum
    return energy >= minimum


def _unit_is_using_any_ability(unit: "Unit", abilities: FrozenSet) -> bool:
    if not abilities or unit is None:
        return False
    checker = getattr(unit, "is_using_ability", None)
    if not callable(checker):
        return False
    try:
        return checker(abilities) is True
    except Exception:
        return False


def _spell_store(bot: Any, family: str) -> Optional[dict]:
    attr = _SPELL_CD_ATTR.get(family)
    if bot is None or attr is None:
        return None
    store = getattr(bot, attr, None)
    return store if isinstance(store, dict) else None


def _spell_ability_on_cooldown(unit: "Unit", bot: Any, family: str) -> bool:
    if bot is None or unit is None:
        return False
    try:
        tag = int(getattr(unit, "tag", 0) or 0)
    except (TypeError, ValueError):
        return False
    if tag <= 0:
        return False
    ready_after = _spell_store(bot, family)
    if ready_after is None:
        return False
    try:
        until = int(ready_after.get(tag, 0) or 0)
    except (TypeError, ValueError):
        return False
    if until <= 0:
        return False
    return _unit_game_loop(bot, unit) < until


def mark_spell_ability_cooldown(bot: Any, unit: "Unit", family: str) -> None:
    """Stamp a local ready-at game loop after we issue a spell.

    Used when ``query_spell_abilities`` is off (the CTF default). Seconds come
    from the Faster-speed table for that family; energy-only spells use 1s so
    the mask stays closed until energy updates. Reissue must pass
    ``stamp_cooldown=False`` so leftover orders do not reset the timer.
    """
    if not _SC2_AVAILABLE or bot is None or unit is None:
        return
    table = _SPELL_CD_SECONDS.get(family)
    attr = _SPELL_CD_ATTR.get(family)
    if not table or attr is None:
        return
    tid = getattr(unit, "type_id", None)
    if tid not in table:
        return
    try:
        tag = int(getattr(unit, "tag", 0) or 0)
    except (TypeError, ValueError):
        return
    if tag <= 0:
        return
    secs = float(table.get(tid, 1.0))
    loops = int(math.ceil(secs * GAME_LOOPS_PER_SECOND))
    ready = _unit_game_loop(bot, unit) + max(loops, 1)
    store = getattr(bot, attr, None)
    if not isinstance(store, dict):
        setattr(bot, attr, {})
        store = getattr(bot, attr)
    store[tag] = ready


def _heal_ability_on_cooldown(unit: "Unit", bot: Any) -> bool:
    return _spell_ability_on_cooldown(unit, bot, "heal")


def mark_heal_ability_cooldown(bot: Any, unit: "Unit") -> None:
    """Best-effort Transfuse/Heal cooldown stamp when SC2 ability query is unavailable."""
    mark_spell_ability_cooldown(bot, unit, "heal")


def _owner_can_afford_repair(bot: Any, unit: "Unit") -> bool:
    if bot is None:
        return True
    try:
        owner_id = int(getattr(unit, "owner_id", 0) or 0)
    except (TypeError, ValueError):
        return True
    client_pid = int(getattr(bot, "player_id", 1) or 1)
    if owner_id == client_pid:
        try:
            return int(getattr(bot, "minerals", 0) or 0) >= _SCV_REPAIR_MIN_MINERALS
        except (TypeError, ValueError):
            return False
    # CTF heal mode grants P2 ``debug_free``; no per-owner mineral obs on the bot.
    return True


async def refresh_healer_ability_cache(
    bot: Any,
    *,
    include_heal: bool = True,
    include_aoe: bool = True,
    include_snipe: bool = True,
    units: Optional[Sequence[Any]] = None,
) -> None:
    """Refresh ``bot._available_abilities_by_tag`` for healers / SCVs / AoE / snipe.

    One SC2 ``query_available_abilities_with_tag`` with
    ``ignore_resource_requirements=True`` (cooldown only). Energy, minerals, and
    local cooldown stamps stay in ``unit_can_heal`` / ``unit_can_repair`` /
    ``unit_can_aoe`` / ``unit_can_snipe``. Double-gating energy through SC2 caused
    false negatives (AoE permanently closed while cloak stayed open).

    CTF does **not** call this every step by default (see
    ``env.query_spell_abilities``). When skipped, ``unit_can_*`` uses type +
    energy + upgrade only. Attack ``weapon_cooldown`` is unrelated — it comes
    from the observation proto.

    ``include_heal`` / ``include_aoe`` / ``include_snipe`` skip matching unit
    types when those command heads are not configured. Overlap types (Ghost,
    Infestor, HT) are queried when any relevant flag is true. Pass ``units``
    (living pool combat units) to avoid scanning every ``bot.all_units`` entry.
    Results are stored as frozensets so mask hot paths do not re-freeze every
    ``unit_can_*`` call.
    """
    if bot is None or not _SC2_AVAILABLE:
        return
    if not (include_heal or include_aoe or include_snipe):
        bot._available_abilities_by_tag = {}
        return
    client = getattr(bot, "client", None)
    query = getattr(client, "query_available_abilities_with_tag", None)
    if client is None or not callable(query):
        return
    source = units if units is not None else (getattr(bot, "all_units", []) or [])
    query_units = []
    seen_tags: set = set()
    for unit in source:
        if unit is None:
            continue
        tid = getattr(unit, "type_id", None)
        need = False
        if include_heal and (tid in _HEAL_ABILITIES_BY_TYPE or tid == UnitTypeId.SCV):
            need = True
        if include_aoe and tid in _AOE_ENERGY_COST:
            need = True
        if include_snipe and tid in _SNIPE_ENERGY_COST:
            need = True
        if not need:
            continue
        tag = int(getattr(unit, "tag", 0) or 0)
        if tag and tag not in seen_tags:
            seen_tags.add(tag)
            query_units.append(unit)
    if not query_units:
        bot._available_abilities_by_tag = {}
        return
    try:
        raw = await query(query_units, ignore_resource_requirements=True)
        cache = raw if isinstance(raw, dict) else {}
        bot._available_abilities_by_tag = _freeze_ability_map(cache)
    except Exception:
        return


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


def unit_can_cloak(unit: "Unit", upgrades: frozenset | None = None) -> bool:
    """True when the unit can activate personal/field cloak this step.

    Already-cloaked units return False. Ghost / Banshee need their cloak tech
    and activation energy. Mothership field cloak is energy-gated only.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    if bool(getattr(unit, "is_cloaked", False)):
        return False
    tid = getattr(unit, "type_id", None)
    upgrades = upgrades or frozenset()
    energy = float(getattr(unit, "energy", 0.0) or 0.0)
    required = _CLOAK_REQUIRED_UPGRADE.get(tid)
    if required is not None:
        if required not in upgrades:
            return False
        return energy >= _CLOAK_ACTIVATION_ENERGY
    if tid == UnitTypeId.MOTHERSHIP:
        return energy >= _MOTHERSHIP_CLOAK_ENERGY
    return False


def unit_can_burrow(unit: "Unit", upgrades: frozenset | None = None) -> bool:
    """True when the unit can burrow this step (not already burrowed).

    Most Zerg need ``BURROW`` researched. Widow Mine and Lurker burrow without
    that tech.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    if tid not in _BURROW_DOWN_TYPES:
        return False
    if bool(getattr(unit, "is_burrowed", False)):
        return False
    if tid in _BURROW_WITHOUT_TECH:
        return True
    return UpgradeId.BURROW in (upgrades or frozenset())


def unit_can_conceal(unit: "Unit", upgrades: frozenset | None = None) -> bool:
    """True when ``cloak_burrow`` can do useful work on this unit."""
    return unit_can_cloak(unit, upgrades) or unit_can_burrow(unit, upgrades)


def unit_can_uncloak(unit: "Unit") -> bool:
    """True when a toggle-cloak unit is currently cloaked and can reveal."""
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    if tid not in {UnitTypeId.GHOST, UnitTypeId.BANSHEE}:
        return False
    return bool(getattr(unit, "is_cloaked", False))


def unit_can_unburrow(unit: "Unit") -> bool:
    """True when the unit is in a burrowed morph that can surface."""
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    if tid in _BURROW_UP_TYPES:
        return True
    return bool(getattr(unit, "is_burrowed", False)) and tid in _BURROW_DOWN_TYPES


def unit_can_reveal(unit: "Unit") -> bool:
    """True when ``uncloak_unburrow`` can do useful work on this unit."""
    return unit_can_uncloak(unit) or unit_can_unburrow(unit)


def unit_can_repair(unit: "Unit", bot: Any = None) -> bool:
    """True when an SCV can start or continue Repair (minerals + not on CD).

    Prefer SC2 ``query_available_abilities`` cached on ``bot`` (includes mineral
    cost and cooldown). Falls back to owner minerals and channel state.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    if getattr(unit, "type_id", None) != UnitTypeId.SCV:
        return False
    abilities = _available_abilities_for_unit(unit, bot)
    if abilities is not None:
        return bool(_REPAIR_ABILITIES & abilities)
    if not _owner_can_afford_repair(bot, unit):
        return False
    return not _unit_is_using_any_ability(unit, _REPAIR_CHANNEL_ABILITIES)


def unit_can_heal(unit: "Unit", bot: Any = None) -> bool:
    """True when a non-SCV healer can cast heal this step.

    Medivac / Queen / Shield Battery. SCV uses ``unit_can_repair`` instead.
    Requires sufficient energy, the local Transfuse/Heal stamp, and not already
    channeling. Uses ``bot._available_abilities_by_tag`` only when the env
    opted into ``query_spell_abilities``.
    """
    if not _SC2_AVAILABLE or unit is None:
        return False
    tid = getattr(unit, "type_id", None)
    required = _HEAL_ABILITIES_BY_TYPE.get(tid)
    if not required:
        return False
    if not _heal_energy_sufficient(unit):
        return False
    if _heal_ability_on_cooldown(unit, bot):
        return False
    if _unit_is_using_any_ability(unit, _HEAL_CHANNEL_ABILITIES):
        return False
    abilities = _available_abilities_for_unit(unit, bot)
    if abilities is not None:
        return bool(required & abilities)
    return True


def unit_can_cast_heal(unit: "Unit", bot: Any = None) -> bool:
    """True when *unit* can start a heal/repair/transfuse right now (either family)."""
    return unit_can_heal(unit, bot) or unit_can_repair(unit, bot)


def heal_target_is_eligible(healer: "Unit", ally: "Unit") -> bool:
    """True when ``ally``'s biology matches what ``healer``'s ability can affect.

    Real SC2 silently ignores an order targeting the wrong category — Medivac
    Heal and Queen Transfuse only affect Biological units, SCV Repair only
    Mechanical ones. Without this check a Medivac clicked onto a wounded Siege
    Tank resolves a target and credits a cast, but the in-game order never heals.
    """
    if not _SC2_AVAILABLE or healer is None or ally is None:
        return True
    tid = getattr(healer, "type_id", None)
    if tid in (UnitTypeId.MEDIVAC, UnitTypeId.QUEEN):
        return bool(getattr(ally, "is_biological", False))
    if tid == UnitTypeId.SCV:
        return bool(getattr(ally, "is_mechanical", False))
    return True


def unit_is_wounded(unit: "Unit") -> bool:
    """True when *unit* is missing health (shields excluded — heals restore HP)."""
    if unit is None:
        return False
    try:
        health = float(getattr(unit, "health", 0.0) or 0.0)
        health_max = float(getattr(unit, "health_max", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if health_max <= 0.0:
        return False
    return health < health_max - 1e-6


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


def unit_is_loadable_passenger(unit: "Unit") -> bool:
    """Non-structure, non-transport with a positive embark cost."""
    if unit is None or bool(getattr(unit, "is_structure", False)):
        return False
    if unit_is_transport(unit):
        return False
    return float(getattr(unit, "cargo_size", 0) or 0) > 0


def unit_transport_free_slots(unit: "Unit") -> int:
    cargo_max = int(getattr(unit, "cargo_max", 0) or 0)
    if cargo_max <= 0:
        return 0
    used = int(getattr(unit, "cargo_used", 0) or 0)
    return max(cargo_max - used, 0)


def load_pair_is_eligible(actor: "Unit", target: "Unit") -> bool:
    """Whether ``actor`` can load/board ``target`` (dual-role transport↔passenger).

    Transport actors issue ``LOAD`` on a passenger; passenger actors smart onto a
    transport. Cargo size must fit the transport's free slots.
    """
    if actor is None or target is None:
        return False
    if int(getattr(actor, "tag", 0) or 0) == int(getattr(target, "tag", 0) or 0):
        return False
    if int(getattr(actor, "owner_id", 0) or 0) != int(
        getattr(target, "owner_id", 0) or 0
    ):
        return False
    if bool(getattr(actor, "is_structure", False)) or bool(
        getattr(target, "is_structure", False)
    ):
        return False

    actor_is_transport = unit_is_transport(actor)
    target_is_transport = unit_is_transport(target)
    if actor_is_transport and unit_is_loadable_passenger(target):
        size = int(getattr(target, "cargo_size", 0) or 0)
        return unit_transport_free_slots(actor) >= size > 0
    if (not actor_is_transport) and target_is_transport and unit_is_loadable_passenger(
        actor
    ):
        size = int(getattr(actor, "cargo_size", 0) or 0)
        return unit_transport_free_slots(target) >= size > 0
    return False


def unit_can_unload(unit: "Unit") -> bool:
    if unit is None:
        return False
    return int(getattr(unit, "cargo_max", 0) or 0) > 0 and int(
        getattr(unit, "cargo_used", 0) or 0
    ) > 0


# Half-width of the 32×32 tactical crop in game units (matches CTF / Level7 bots).
TACTICAL_FOVEA_MAX_DELTA = 17.0


def unit_weapon_ready(unit: "Unit") -> bool:
    """True when the unit's weapon can fire this step.

    Prefers an explicit ``weapon_ready`` bool. Falls back to ``weapon_cooldown``
    (ready when ≤ 0). Missing cooldown info is treated as ready so unarmed /
    mock units without the attribute do not spuriously block attacks.
    """
    if unit is None:
        return False
    wr = getattr(unit, "weapon_ready", None)
    if isinstance(wr, bool):
        return wr
    try:
        cd = float(getattr(unit, "weapon_cooldown", 0.0) or 0.0)
    except (TypeError, ValueError):
        return True
    return cd <= 1e-6


def unit_has_weapon(unit: "Unit") -> bool:
    """True when the unit has a positive ground or air DPS."""
    if unit is None:
        return False
    try:
        ground = float(getattr(unit, "ground_dps", 0.0) or 0.0)
        air = float(getattr(unit, "air_dps", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return ground > 0.0 or air > 0.0


def _unit_is_burrowed(unit: "Unit") -> bool:
    """True when the unit is currently burrowed (flag or BURROWED type name)."""
    if unit is None:
        return False
    if bool(getattr(unit, "is_burrowed", False)):
        return True
    tid = getattr(unit, "type_id", None)
    return tid is not None and "BURROWED" in str(getattr(tid, "name", "")).upper()


def unit_can_attack(
    unit: "Unit", upgrades: frozenset | None = None
) -> bool:
    """True when this unit can usefully receive an attack order this step.

    Requires a ready weapon, **or** a weaponless unit whose policy ``attack``
    head maps to a combat ability (Swarm Host → Spawn Locusts, Disruptor →
    Purification Nova, Nova ball → move). Burrowed units that cannot fire
    (Zergling, Roach, …) are excluded even if Tunneling Claws makes them
    mobile. Lurkers, sieged tanks, Liberator AG, and burrowed Widow Mines
    still count — they attack from that planted morph. Burrowed Swarm Hosts
    stay legal: Spawn Locusts works while burrowed.
    """
    # Ability-mapped attackers have no DPS; gate them before the weapon check
    # so attack_local / attack_unit stay open for Swarm Host-only squads.
    if uses_spawn_locusts_for_attack(unit):
        return True
    if uses_purification_nova_for_attack(unit):
        return True
    if is_purification_nova_ball(unit):
        return True
    if not unit_has_weapon(unit) or not unit_weapon_ready(unit):
        return False
    tid = getattr(unit, "type_id", None)
    if tid in _ATTACKS_WHILE_IMMOBILE:
        return True
    if tid in _ATTACK_ONLY_WHEN_BURROWED and not _unit_is_burrowed(unit):
        return False
    if _unit_is_burrowed(unit):
        return False
    if unit_is_immobile(unit, upgrades):
        return False
    return True


def unit_in_tactical_fovea(
    focal: "Unit",
    other: "Unit",
    *,
    half: float = TACTICAL_FOVEA_MAX_DELTA,
) -> bool:
    """True if ``other`` lies inside the 32×32 tactical crop centered on ``focal``."""
    if focal is None or other is None:
        return False
    try:
        fx = float(focal.position.x)
        fy = float(focal.position.y)
        ox = float(other.position.x)
        oy = float(other.position.y)
    except (TypeError, ValueError, AttributeError):
        return False
    h = float(half)
    return abs(ox - fx) <= h and abs(oy - fy) <= h

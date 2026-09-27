"""Command-dependent actor/target legality for the CTF squad pointer stack.

The policy samples ``command → actors → target``, so actor and target legality
depend on the sampled command and cannot live in the flat env action mask (which
is built before the command exists). These arrays ride in the observation
instead, which means the learner narrows legality with exactly the same values
the collector used and PPO probabilities stay consistent.

Every array is indexed by the same stable slots the observation uses:
allied arrays by squad slot (sorted allied tags), enemy arrays by visible-enemy
slot (sorted hostile tags).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.agent.squad_constants import MAX_SQUAD_ENEMIES, MAX_SQUAD_UNITS
from src.utils.unit_feature_helpers import (
    heal_target_is_eligible,
    load_pair_is_eligible,
    snipe_target_is_eligible,
    unit_can_aoe,
    unit_can_attack,
    unit_can_cast_heal,
    unit_can_conceal,
    unit_can_load,
    unit_can_reveal,
    unit_can_snipe,
    unit_can_unload,
    unit_in_tactical_fovea,
    unit_is_immobile,
    unit_is_wounded,
)

#: Observation keys produced by :func:`build_squad_target_masks`.
SQUAD_TARGET_MASK_KEYS = (
    "heal_actor_mask",
    "heal_target_mask",
    "load_actor_mask",
    "load_target_mask",
    "attack_actor_mask",
    "aoe_actor_mask",
    "snipe_actor_mask",
    "unload_actor_mask",
    "cloak_actor_mask",
    "uncloak_actor_mask",
    "move_actor_mask",
)


def _is_controllable(unit: Any) -> bool:
    """Structures and embarked passengers cannot receive their own orders."""
    if unit is None:
        return False
    if bool(getattr(unit, "is_structure", False)):
        return False
    return True


def zero_target_masks() -> Dict[str, np.ndarray]:
    """All-zero legality arrays (used for the zero observation)."""
    return {key: np.zeros(MAX_SQUAD_UNITS, dtype=np.float32) for key in SQUAD_TARGET_MASK_KEYS}


def build_squad_target_masks(
    allies: Sequence[Optional[Any]],
    *,
    bot: Any = None,
    enemy_race: Optional[str] = None,
    upgrades: Optional[frozenset] = None,
    has_tactical_air_enemy: bool = False,
) -> Dict[str, np.ndarray]:
    """Actor/target legality arrays for the allied squad slots.

    ``allies`` is slot-aligned with ``unit_scalars``: index *i* holds the unit in
    squad slot *i*, or ``None`` for a dead/padded slot.

    ``heal_target_mask`` marks wounded allies that **some** legal healer in this
    squad can actually affect. Per-caster biology is re-checked at actuation, so
    a mixed Medivac + SCV squad cannot be tricked into a silent no-op order.

    ``load_target_mask`` marks allies that **some** load-capable actor can pair
    with (transport↔passenger, cargo must fit). Actuation re-checks the pair.

    ``attack_actor_mask`` marks units that can usefully receive an attack order
    (weapon ready, or weaponless casters whose attack remaps — Swarm Host /
    Disruptor / Nova ball; not a non-lurker burrowed / immobile morph). Empty ⇒
    both ``attack_unit`` and ``attack_local`` are masked.

    ``aoe_actor_mask`` marks ready AoE casters (energy, research, ability off
    cooldown). Viper also needs ``has_tactical_air_enemy`` (air near any ally).
    Command gating for ``aoe_local`` additionally requires
    ``engage_enemy_mask`` (hostile within fovea of any living ally).

    ``snipe_actor_mask`` marks ready unit-target special casters (Ghost Snipe,
    Yamato, Neural Parasite, Feedback). Command gating for ``snipe`` also needs
    ``snipe_target_mask`` (an eligible hostile for *some* ready caster).

    ``move_actor_mask`` marks controllable units that can walk (not sieged /
    burrowed immobile). Used to narrow move actors; the move command stays open.

    ``unload_actor_mask`` / ``cloak_actor_mask`` / ``uncloak_actor_mask`` gate
    the matching command heads when no capable unit is present.
    """
    heal_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    heal_target = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    load_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    load_target = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    attack_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    aoe_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    snipe_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unload_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    cloak_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    uncloak_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    move_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)

    slots: List[Optional[Any]] = list(allies)[:MAX_SQUAD_UNITS]
    healers: List[Any] = []
    load_candidates: List[tuple] = []
    upg = upgrades if upgrades is not None else frozenset()
    for slot, unit in enumerate(slots):
        if unit is None:
            continue
        controllable = _is_controllable(unit)
        if controllable and unit_can_cast_heal(unit, bot):
            heal_actor[slot] = 1.0
            healers.append(unit)
        if controllable and unit_can_load(unit):
            load_candidates.append((slot, unit))
        if controllable and unit_can_attack(unit, upg):
            attack_actor[slot] = 1.0
        if controllable and unit_can_aoe(
            unit,
            enemy_race=enemy_race,
            upgrades=upg,
            bot=bot,
            has_tactical_air_enemy=has_tactical_air_enemy,
        ):
            aoe_actor[slot] = 1.0
        if controllable and unit_can_snipe(unit, upgrades=upg, bot=bot):
            snipe_actor[slot] = 1.0
        if controllable and unit_can_unload(unit):
            unload_actor[slot] = 1.0
        if controllable and unit_can_conceal(unit, upg):
            cloak_actor[slot] = 1.0
        if controllable and unit_can_reveal(unit):
            uncloak_actor[slot] = 1.0
        if controllable and not unit_is_immobile(unit, upg):
            move_actor[slot] = 1.0

    if healers:
        for slot, unit in enumerate(slots):
            if unit is None or not unit_is_wounded(unit):
                continue
            unit_tag = getattr(unit, "tag", None)
            for healer in healers:
                if getattr(healer, "tag", None) == unit_tag:
                    continue
                if heal_target_is_eligible(healer, unit):
                    heal_target[slot] = 1.0
                    break

    for actor_slot, actor in load_candidates:
        for target_slot, target in enumerate(slots):
            if target is None:
                continue
            if load_pair_is_eligible(actor, target):
                load_actor[actor_slot] = 1.0
                load_target[target_slot] = 1.0

    return {
        "heal_actor_mask": heal_actor,
        "heal_target_mask": heal_target,
        "load_actor_mask": load_actor,
        "load_target_mask": load_target,
        "attack_actor_mask": attack_actor,
        "aoe_actor_mask": aoe_actor,
        "snipe_actor_mask": snipe_actor,
        "unload_actor_mask": unload_actor,
        "cloak_actor_mask": cloak_actor,
        "uncloak_actor_mask": uncloak_actor,
        "move_actor_mask": move_actor,
    }


def build_enemy_target_mask(enemies: Sequence[Optional[Any]]) -> np.ndarray:
    """Legality of each visible-enemy slot as an ``attack_unit`` target."""
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    for slot, unit in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if unit is None:
            continue
        mask[slot] = 1.0
    return mask


def build_snipe_target_mask(
    enemies: Sequence[Optional[Any]],
    allies: Sequence[Optional[Any]],
    *,
    bot: Any = None,
    upgrades: Optional[frozenset] = None,
) -> np.ndarray:
    """Enemy slots that **some** ready snipe caster in this squad can legally hit.

    Per-caster biology / energy-bar rules are re-checked at actuation, matching
    the heal_target_mask pattern for mixed squads.
    """
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    upg = upgrades if upgrades is not None else frozenset()
    casters = [
        u
        for u in list(allies)[:MAX_SQUAD_UNITS]
        if u is not None and _is_controllable(u) and unit_can_snipe(u, upgrades=upg, bot=bot)
    ]
    if not casters:
        return mask
    for slot, enemy in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if enemy is None:
            continue
        for caster in casters:
            if snipe_target_is_eligible(caster, enemy):
                mask[slot] = 1.0
                break
    return mask


def _mark_extra_hostile_presence(
    mask: np.ndarray,
    hostiles: Sequence[Optional[Any]],
    foci: Sequence[Any],
) -> None:
    """OR a presence bit when a non-pointer hostile (building) is in any fovea.

    Command gating only reads ``mask.any()``. Pointer-aligned mobile slots stay
    unchanged when a mobile is already marked.
    """
    if mask.any() or not hostiles or not foci:
        return
    for hostile in hostiles:
        if hostile is None:
            continue
        for focus in foci:
            if unit_in_tactical_fovea(focus, hostile):
                mask[0] = 1.0
                return


def build_tactical_enemy_mask(
    enemies: Sequence[Optional[Any]],
    anchor: Optional[Any],
    *,
    air_only: bool = False,
    extra_hostiles: Sequence[Optional[Any]] | None = None,
) -> np.ndarray:
    """Enemy-pointer slots that lie inside the hard-anchor's tactical fovea.

    Matches the ``anchor_tactical`` crop the policy sees. Prefer
    :func:`build_engage_enemy_mask` for **command** gating: actuation decodes
    spatial clicks in the *newly selected* anchor frame, so last-step crop
    emptiness must not permanently close ``aoe_local`` / ``attack_local``.

    ``extra_hostiles`` (enemy buildings) open a presence bit so spatial attack
    stays legal when the only nearby target is a structure. Ignored for
    ``air_only`` (Viper bomb).
    """
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    if anchor is None:
        return mask
    for slot, unit in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if unit is None:
            continue
        if air_only and not bool(getattr(unit, "is_flying", False)):
            continue
        if unit_in_tactical_fovea(anchor, unit):
            mask[slot] = 1.0
    if not air_only:
        _mark_extra_hostile_presence(mask, extra_hostiles or (), (anchor,))
    return mask


def build_engage_enemy_mask(
    enemies: Sequence[Optional[Any]],
    allies: Sequence[Optional[Any]],
    *,
    air_only: bool = False,
    extra_hostiles: Sequence[Optional[Any]] | None = None,
) -> np.ndarray:
    """Enemy slots within tactical half-width of **any** living allied unit.

    Used to open ``attack_local`` / ``aoe_local`` when some selectable ally
    could become this step's hard-anchor and still see the hostile (mobile or
    building). ``air_only`` stays flying mobiles only.
    """
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    foci = [u for u in list(allies)[:MAX_SQUAD_UNITS] if u is not None]
    if not foci:
        return mask
    for slot, enemy in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if enemy is None:
            continue
        if air_only and not bool(getattr(enemy, "is_flying", False)):
            continue
        for ally in foci:
            if unit_in_tactical_fovea(ally, enemy):
                mask[slot] = 1.0
                break
    if not air_only:
        _mark_extra_hostile_presence(mask, extra_hostiles or (), foci)
    return mask

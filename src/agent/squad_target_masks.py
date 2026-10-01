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
    unit_can_burrow,
    unit_can_cloak,
    unit_can_heal,
    unit_can_load,
    unit_can_repair,
    unit_can_reveal,
    unit_can_snipe,
    unit_can_unload,
    unit_in_tactical_fovea,
    unit_is_immobile,
    neural_spell_role,
    snipe_same_owner_blocked,
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

#: Per-slot readiness reused by unit scalars (not part of the obs mask dict).
SQUAD_ACTOR_READINESS_KEYS = (
    "can_heal",
    "can_repair",
    "can_aoe",
    "can_snipe",
    "can_load",
    "can_unload",
    "can_cloak",
    "is_immobile",
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


def zero_actor_readiness() -> Dict[str, np.ndarray]:
    """All-zero per-slot readiness (mirrors :data:`SQUAD_ACTOR_READINESS_KEYS`)."""
    return {
        key: np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
        for key in SQUAD_ACTOR_READINESS_KEYS
    }


def build_squad_legality(
    allies: Sequence[Optional[Any]],
    *,
    bot: Any = None,
    enemy_race: Optional[str] = None,
    upgrades: Optional[frozenset] = None,
    has_tactical_air_enemy: bool = False,
    compute_heal: bool = True,
    compute_load: bool = True,
    compute_attack: bool = True,
    compute_aoe: bool = True,
    compute_snipe: bool = True,
    compute_unload: bool = True,
    compute_cloak: bool = True,
    compute_uncloak: bool = True,
    compute_move: bool = True,
    perspective_player_id: Optional[int] = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Actor/target masks plus per-slot readiness for unit scalars.

    One ``unit_can_*`` pass fills both products so
    :class:`~src.transformers.squad_observation_builder.SquadObservationBuilder`
    does not re-run the same readiness helpers when writing ``can_heal`` /
    ``can_aoe`` / ``can_snipe`` / cargo / cloak scalars.
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
    readiness = zero_actor_readiness()

    slots: List[Optional[Any]] = list(allies)[:MAX_SQUAD_UNITS]
    healers: List[Any] = []
    load_candidates: List[tuple] = []
    upg = upgrades if upgrades is not None else frozenset()
    for slot, unit in enumerate(slots):
        if unit is None:
            continue

        # Scalar readiness for every living ally slot (structures included).
        is_immobile = unit_is_immobile(unit, upg)
        readiness["is_immobile"][slot] = float(is_immobile)
        can_cloak = unit_can_cloak(unit, upg)
        readiness["can_cloak"][slot] = float(can_cloak)
        can_load = unit_can_load(unit)
        readiness["can_load"][slot] = float(can_load)
        can_unload = unit_can_unload(unit)
        readiness["can_unload"][slot] = float(can_unload)

        can_heal = False
        can_repair = False
        if compute_heal:
            can_heal = unit_can_heal(unit, bot)
            can_repair = unit_can_repair(unit, bot)
            readiness["can_heal"][slot] = float(can_heal)
            readiness["can_repair"][slot] = float(can_repair)

        can_aoe = False
        if compute_aoe:
            can_aoe = unit_can_aoe(
                unit,
                enemy_race=enemy_race,
                upgrades=upg,
                bot=bot,
                has_tactical_air_enemy=has_tactical_air_enemy,
            )
            readiness["can_aoe"][slot] = float(can_aoe)

        can_snipe = False
        if compute_snipe:
            can_snipe = unit_can_snipe(unit, upgrades=upg, bot=bot)
            readiness["can_snipe"][slot] = float(can_snipe)

        # Neural Parasite: the live owner borrows aoe/snipe only. The original
        # squad keeps the slot for move/attack but must not spend the spell.
        spell_role = neural_spell_role(unit, perspective_player_id)
        if spell_role == "stolen_away":
            readiness["can_aoe"][slot] = 0.0
            readiness["can_snipe"][slot] = 0.0

        controllable = _is_controllable(unit)
        if not controllable:
            continue

        borrowed = spell_role == "borrowed"
        if not borrowed:
            if compute_heal and (can_heal or can_repair):
                heal_actor[slot] = 1.0
                healers.append(unit)
            if compute_load and can_load:
                load_candidates.append((slot, unit))
            if compute_attack and unit_can_attack(unit, upg):
                attack_actor[slot] = 1.0
            if compute_unload and can_unload:
                unload_actor[slot] = 1.0
            if compute_cloak and (can_cloak or unit_can_burrow(unit, upg)):
                cloak_actor[slot] = 1.0
            if compute_uncloak and unit_can_reveal(unit):
                uncloak_actor[slot] = 1.0
            if compute_move and not is_immobile:
                move_actor[slot] = 1.0
        if spell_role != "stolen_away":
            if compute_aoe and can_aoe:
                aoe_actor[slot] = 1.0
            if compute_snipe and can_snipe:
                snipe_actor[slot] = 1.0

    if compute_heal and healers:
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

    masks = {
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
    return masks, readiness


def build_squad_target_masks(
    allies: Sequence[Optional[Any]],
    *,
    bot: Any = None,
    enemy_race: Optional[str] = None,
    upgrades: Optional[frozenset] = None,
    has_tactical_air_enemy: bool = False,
    compute_heal: bool = True,
    compute_load: bool = True,
    compute_attack: bool = True,
    compute_aoe: bool = True,
    compute_snipe: bool = True,
    compute_unload: bool = True,
    compute_cloak: bool = True,
    compute_uncloak: bool = True,
    compute_move: bool = True,
    perspective_player_id: Optional[int] = None,
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

    Prefer :func:`build_squad_legality` when unit scalars also need readiness —
    it returns ``(masks, readiness)`` from the same ``unit_can_*`` pass.
    """
    masks, _readiness = build_squad_legality(
        allies,
        bot=bot,
        enemy_race=enemy_race,
        upgrades=upgrades,
        has_tactical_air_enemy=has_tactical_air_enemy,
        compute_heal=compute_heal,
        compute_load=compute_load,
        compute_attack=compute_attack,
        compute_aoe=compute_aoe,
        compute_snipe=compute_snipe,
        compute_unload=compute_unload,
        compute_cloak=compute_cloak,
        compute_uncloak=compute_uncloak,
        compute_move=compute_move,
        perspective_player_id=perspective_player_id,
    )
    return masks


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
    casters: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Enemy slots that **some** ready snipe caster in this squad can legally hit.

    Per-caster biology / energy-bar rules are re-checked at actuation, matching
    the heal_target_mask pattern for mixed squads.

    Pass ``casters`` (already-ready units from ``snipe_actor_mask``) to avoid a
    second ``unit_can_snipe`` pass over the squad.
    """
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    if casters is None:
        upg = upgrades if upgrades is not None else frozenset()
        ready = [
            u
            for u in list(allies)[:MAX_SQUAD_UNITS]
            if u is not None
            and _is_controllable(u)
            and unit_can_snipe(u, upgrades=upg, bot=bot)
        ]
    else:
        ready = [u for u in casters if u is not None]
    if not ready:
        return mask
    for slot, enemy in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if enemy is None:
            continue
        for caster in ready:
            if not snipe_target_is_eligible(caster, enemy):
                continue
            if snipe_same_owner_blocked(caster, enemy):
                continue
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


def build_engage_enemy_masks(
    enemies: Sequence[Optional[Any]],
    allies: Sequence[Optional[Any]],
    *,
    extra_hostiles: Sequence[Optional[Any]] | None = None,
) -> tuple[np.ndarray, bool]:
    """Engage fovea mask plus whether any **flying mobile** is in that fovea.

    One ally×enemy pass serves both ``engage_enemy_mask`` (command gating) and
    Viper's ``has_tactical_air_enemy`` flag, avoiding a second fovea scan.
    """
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    has_air = False
    foci = [u for u in list(allies)[:MAX_SQUAD_UNITS] if u is not None]
    if not foci:
        return mask, False
    for slot, enemy in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if enemy is None:
            continue
        for ally in foci:
            if unit_in_tactical_fovea(ally, enemy):
                mask[slot] = 1.0
                if bool(getattr(enemy, "is_flying", False)):
                    has_air = True
                break
    _mark_extra_hostile_presence(mask, extra_hostiles or (), foci)
    return mask, has_air


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
    if not air_only:
        mask, _ = build_engage_enemy_masks(
            enemies, allies, extra_hostiles=extra_hostiles
        )
        return mask
    mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    foci = [u for u in list(allies)[:MAX_SQUAD_UNITS] if u is not None]
    if not foci:
        return mask
    for slot, enemy in enumerate(list(enemies)[:MAX_SQUAD_ENEMIES]):
        if enemy is None:
            continue
        if not bool(getattr(enemy, "is_flying", False)):
            continue
        for ally in foci:
            if unit_in_tactical_fovea(ally, enemy):
                mask[slot] = 1.0
                break
    return mask

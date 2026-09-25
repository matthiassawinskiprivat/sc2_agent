"""Squad flat action mask: actor padding and command prerequisite gating.

Only the actor and command slices carry information in the flat mask. Legality
that depends on the *sampled* command lives in the observation and is applied
inside ``SquadRecurrentPolicy`` (see ``test_squad_autoregressive_policy.py``).
"""

import numpy as np
import pytest

from src.agent.squad_constants import (
    MAX_SQUAD_ENEMIES,
    MAX_SQUAD_UNITS,
    SQUAD_FLAT_ACTION_DIM,
    SQUAD_MASK_COMMAND_OFFSET,
    squad_action_space_nvec,
)
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

HEADS = [
    "move_global",
    "attack_local",
    "attack_unit",
    "heal_unit",
    "load_unit",
    "aoe_local",
    "unload_local",
]
CATS = ["cloak_burrow", "uncloak_unburrow"]
CMD = SQUAD_MASK_COMMAND_OFFSET


def _bot(heads=None, categorical=None):
    from src.agent.ctf_bot import CaptureTheFlagBot

    bot = CaptureTheFlagBot.__new__(CaptureTheFlagBot)
    bot.training_conf = {
        "action_heads": list(heads if heads is not None else HEADS),
        "categorical_heads": list(categorical if categorical is not None else CATS),
    }
    return bot


def _alive(n=2):
    unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unit_mask[:n] = 1.0
    return unit_mask


def _obs(
    *,
    healers=0,
    patients=0,
    loaders=0,
    load_targets=0,
    enemies=0,
    tactical_enemies=None,
    engage_enemies=None,
    attack_ready=None,
    aoe=0,
    unload=0,
    cloak=0,
    uncloak=0,
    n_alive=2,
):
    def ally(count):
        arr = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
        arr[:count] = 1.0
        return arr

    enemy = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    enemy[:enemies] = 1.0
    if tactical_enemies is None:
        tactical_enemies = enemies
    if engage_enemies is None:
        # Engage defaults to tactical so older call sites keep their meaning;
        # tests that distinguish hard-anchor vs any-ally set both explicitly.
        engage_enemies = tactical_enemies
    tactical = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    tactical[:tactical_enemies] = 1.0
    engage = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    engage[:engage_enemies] = 1.0
    if attack_ready is None:
        attack_ready = n_alive
    return {
        "unit_mask": _alive(n_alive),
        "heal_actor_mask": ally(healers),
        "heal_target_mask": ally(patients),
        "load_actor_mask": ally(loaders),
        "load_target_mask": ally(load_targets),
        "attack_actor_mask": ally(attack_ready),
        "aoe_actor_mask": ally(aoe),
        "unload_actor_mask": ally(unload),
        "cloak_actor_mask": ally(cloak),
        "uncloak_actor_mask": ally(uncloak),
        "enemy_mask": enemy,
        "tactical_enemy_mask": tactical,
        "engage_enemy_mask": engage,
    }


def test_squad_action_masks_dead_slots_force_not_selected():
    bot = _bot(["move_global", "move_local", "attack_local"], [])
    mask = bot.squad_action_masks(_alive(2))

    assert mask.shape[0] == int(sum(squad_action_space_nvec()))
    assert mask.shape[0] == SQUAD_FLAT_ACTION_DIM
    # Alive slots: both binary options legal
    assert bool(mask[0]) and bool(mask[1])
    assert bool(mask[2]) and bool(mask[3])
    # Dead slots: only action 0 (not selected) legal
    for i in range(2, MAX_SQUAD_UNITS):
        assert bool(mask[i * 2]) is True
        assert bool(mask[i * 2 + 1]) is False

    assert mask[CMD : CMD + 3].all()
    assert not mask[CMD + 3 : CMD + FIXED_UNIVERSAL_COMMAND_DIM].any()


def test_target_and_spatial_slices_are_open_in_the_flat_mask():
    bot = _bot()
    mask = bot.squad_action_masks(_alive(2))
    assert mask[CMD + FIXED_UNIVERSAL_COMMAND_DIM :].all()


def test_heal_unit_masked_without_a_ready_healer():
    bot = _bot()
    obs = _obs(healers=0, patients=1, enemies=1, aoe=1, cloak=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("heal_unit")]) is False
    assert bool(mask[CMD + HEADS.index("attack_unit")]) is True


def test_heal_unit_masked_without_a_wounded_compatible_patient():
    bot = _bot()
    obs = _obs(healers=1, patients=0, enemies=1, aoe=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("heal_unit")]) is False


def test_heal_unit_open_when_healer_and_patient_exist():
    bot = _bot()
    obs = _obs(healers=1, patients=1, enemies=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("heal_unit")]) is True


def test_load_unit_masked_without_a_loader_pair():
    bot = _bot()
    obs = _obs(healers=1, patients=1, loaders=0, load_targets=1, enemies=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("load_unit")]) is False


def test_load_unit_open_when_loader_and_partner_exist():
    bot = _bot()
    obs = _obs(loaders=1, load_targets=1, enemies=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("load_unit")]) is True
    # Heal prerequisites absent: heal stays masked while load opens.
    assert bool(mask[CMD + HEADS.index("heal_unit")]) is False


def test_attack_unit_masked_without_a_visible_hostile():
    bot = _bot()
    obs = _obs(healers=1, patients=1, enemies=0, aoe=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("attack_unit")]) is False
    # No tactical enemy either ⇒ attack_local closes; move stays open.
    assert bool(mask[CMD + HEADS.index("attack_local")]) is False
    assert bool(mask[CMD + HEADS.index("move_global")]) is True


def test_attack_local_masked_without_engage_enemy():
    bot = _bot()
    # Visible on the map but outside every ally's fovea.
    obs = _obs(enemies=1, tactical_enemies=0, engage_enemies=0, attack_ready=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("attack_local")]) is False
    assert bool(mask[CMD + HEADS.index("attack_unit")]) is True


def test_attack_local_open_when_engage_enemy_even_if_hard_anchor_empty():
    """Previous-step hard-anchor may be a distant ling; engage uses any ally."""
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=0, engage_enemies=1, attack_ready=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("attack_local")]) is True


def test_attack_local_open_for_building_in_fovea_without_mobile_enemy():
    """Pylon / bunker in engage fovea opens spatial attack; attack_unit stays army-only."""
    bot = _bot()
    obs = _obs(enemies=0, tactical_enemies=0, engage_enemies=1, attack_ready=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("attack_local")]) is True
    assert bool(mask[CMD + HEADS.index("attack_unit")]) is False


def test_attack_local_open_when_tactical_enemy_and_weapon_ready():
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=1, attack_ready=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("attack_local")]) is True
    assert bool(mask[CMD + HEADS.index("attack_unit")]) is True


def test_both_attack_commands_masked_when_all_weapons_on_cooldown():
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=1, attack_ready=0)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("attack_local")]) is False
    assert bool(mask[CMD + HEADS.index("attack_unit")]) is False
    assert bool(mask[CMD + HEADS.index("move_global")]) is True


def test_aoe_local_masked_without_a_ready_caster():
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=1, aoe=0)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("aoe_local")]) is False


def test_aoe_local_masked_without_engage_enemy():
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=0, engage_enemies=0, aoe=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("aoe_local")]) is False


def test_aoe_local_open_when_engage_enemy_even_if_hard_anchor_empty():
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=0, engage_enemies=1, aoe=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("aoe_local")]) is True


def test_aoe_local_open_when_caster_ready():
    bot = _bot()
    obs = _obs(enemies=1, tactical_enemies=1, aoe=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("aoe_local")]) is True


def test_unload_local_masked_without_loaded_transport():
    bot = _bot()
    obs = _obs(enemies=1, unload=0)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("unload_local")]) is False


def test_unload_local_open_when_transport_has_cargo():
    bot = _bot()
    obs = _obs(enemies=1, unload=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + HEADS.index("unload_local")]) is True


def test_cloak_burrow_masked_without_conceal_capable_unit():
    bot = _bot()
    obs = _obs(enemies=1, cloak=0, uncloak=1)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + len(HEADS) + CATS.index("cloak_burrow")]) is False
    assert bool(mask[CMD + len(HEADS) + CATS.index("uncloak_unburrow")]) is True


def test_uncloak_unburrow_masked_without_concealed_unit():
    bot = _bot()
    obs = _obs(enemies=1, cloak=1, uncloak=0)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + len(HEADS) + CATS.index("cloak_burrow")]) is True
    assert bool(mask[CMD + len(HEADS) + CATS.index("uncloak_unburrow")]) is False


def test_categorical_heads_are_gated_by_conceal_state():
    bot = _bot(["move_global"], ["cloak_burrow", "uncloak_unburrow"])
    obs = _obs(healers=0, patients=0, enemies=0, cloak=0, uncloak=0)
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert bool(mask[CMD + 0]) is True  # move always open
    assert bool(mask[CMD + 1]) is False  # cloak
    assert bool(mask[CMD + 2]) is False  # uncloak


@pytest.mark.parametrize(
    "heads",
    [
        ["attack_unit"],
        ["heal_unit"],
        ["load_unit"],
        ["aoe_local"],
        ["unload_local"],
        ["attack_local"],
    ],
)
def test_command_slice_is_never_fully_masked(heads):
    """A fully-masked command slice would leave the policy nothing to sample."""
    bot = _bot(heads, [])
    obs = _obs(
        healers=0,
        patients=0,
        loaders=0,
        load_targets=0,
        enemies=0,
        tactical_enemies=0,
        attack_ready=0,
    )
    mask = bot.squad_action_masks(obs["unit_mask"], obs)
    assert mask[CMD : CMD + FIXED_UNIVERSAL_COMMAND_DIM].any()

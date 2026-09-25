"""Tests for squad pointer selection resolver."""

import numpy as np

from src.agent.squad_selection import (
    SquadSelectionConfig,
    eligible_actor_mask_from_obs,
    resolve_squad_selection,
    decode_squad_action,
    selected_tags_from_indices,
    resolve_squad_selection_from_binary,
    target_tag_from_slot,
)
from src.agent.squad_constants import (
    CMD_MODE_ALLY_TARGET,
    CMD_MODE_ENEMY_TARGET,
    CMD_MODE_LOAD_TARGET,
    CMD_MODE_NONE,
    CMD_MODE_SPATIAL,
    MAX_SQUAD_UNITS,
    SQUAD_ACTION_COMMAND_IDX,
    SQUAD_ACTION_DIM,
    SQUAD_ACTION_SPATIAL_IDX,
    SQUAD_ACTION_TARGET_IDX,
    SQUAD_FLAT_ACTION_DIM,
    SQUAD_NULL_TARGET_SLOT,
    SQUAD_SPATIAL_CELLS,
    SQUAD_TARGET_SLOTS,
    ally_target_slot,
    command_target_modes,
    enemy_target_slot,
    squad_action_space_nvec,
    target_slot_kind,
    target_slot_local_index,
)
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM as CMD_DIM


def test_resolve_selects_above_threshold():
    logits = np.array([0.0, 2.0, -2.0, 0.0], dtype=np.float32)
    mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    result = resolve_squad_selection(logits, mask, config=SquadSelectionConfig(score_threshold=0.5))
    assert 1 in result.selected_indices
    assert 2 not in result.selected_indices
    assert result.anchor_index == 1


def test_resolve_enforces_min_selected():
    logits = np.array([-5.0, -4.0, -6.0], dtype=np.float32)
    mask = np.ones(3, dtype=np.float32)
    result = resolve_squad_selection(
        logits,
        mask,
        config=SquadSelectionConfig(score_threshold=0.9, min_selected_units=2),
    )
    assert len(result.selected_indices) == 2


def test_binary_selection_min_enforced():
    binary = np.zeros(MAX_SQUAD_UNITS, dtype=np.int64)
    mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    mask[:3] = 1.0
    result = resolve_squad_selection_from_binary(
        binary,
        mask,
        config=SquadSelectionConfig(min_selected_units=1),
    )
    assert len(result.selected_indices) == 1


def test_binary_min_select_stays_inside_eligible_mask():
    """All pointer bits 0 must not backfill a burrowed / incapable slot."""
    binary = np.zeros(MAX_SQUAD_UNITS, dtype=np.int64)
    unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unit_mask[:3] = 1.0
    # Slot 0 = burrowed ling (ineligible), slot 1 = marine (eligible).
    eligible = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    eligible[1] = 1.0
    result = resolve_squad_selection_from_binary(
        binary,
        unit_mask,
        config=SquadSelectionConfig(min_selected_units=1),
        eligible_mask=eligible,
    )
    assert result.selected_indices == (1,)
    assert result.anchor_index == 1


def test_binary_empty_eligible_pool_does_not_backfill():
    binary = np.zeros(MAX_SQUAD_UNITS, dtype=np.int64)
    binary[0] = 1
    unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unit_mask[:2] = 1.0
    eligible = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    result = resolve_squad_selection_from_binary(
        binary,
        unit_mask,
        config=SquadSelectionConfig(min_selected_units=1),
        eligible_mask=eligible,
    )
    assert result.selected_indices == ()
    assert result.anchor_index == 0


def test_decode_squad_action_min_select_uses_obs_attack_mask():
    action = np.zeros(SQUAD_ACTION_DIM, dtype=np.int64)
    # attack_unit at index 2; no actor bits set.
    action[SQUAD_ACTION_COMMAND_IDX] = 2
    unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unit_mask[:2] = 1.0
    attack = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    attack[1] = 1.0
    obs = {
        "command_modes": command_target_modes(
            ["move_global", "attack_local", "attack_unit"], []
        ),
        "attack_actor_mask": attack,
    }
    decoded = decode_squad_action(
        action,
        unit_mask,
        config=SquadSelectionConfig(min_selected_units=1),
        obs=obs,
    )
    assert decoded.command_idx == 2
    assert decoded.selection.selected_indices == (1,)
    assert decoded.selection.anchor_index == 1


def test_eligible_actor_mask_from_obs_reads_command_mode_key():
    attack = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    attack[0] = 1.0
    aoe = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    aoe[2] = 1.0
    obs = {
        "command_modes": command_target_modes(
            ["move_global", "attack_unit", "aoe_local"], []
        ),
        "attack_actor_mask": attack,
        "aoe_actor_mask": aoe,
    }
    assert eligible_actor_mask_from_obs(obs, 1)[0] == 1.0
    assert eligible_actor_mask_from_obs(obs, 2)[2] == 1.0
    assert eligible_actor_mask_from_obs(None, 1) is None


def test_decode_squad_action_layout():
    action = np.zeros(SQUAD_ACTION_DIM, dtype=np.int64)
    action[0] = 1
    action[SQUAD_ACTION_COMMAND_IDX] = 3
    action[SQUAD_ACTION_TARGET_IDX] = enemy_target_slot(2)
    action[SQUAD_ACTION_SPATIAL_IDX] = 42
    mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    mask[0] = 1.0
    decoded = decode_squad_action(action, mask)
    assert 0 in decoded.selection.selected_indices
    assert decoded.command_idx == 3
    assert decoded.target_slot == enemy_target_slot(2)
    assert decoded.spatial_idx == 42


def test_action_space_nvec_matches_flat_mask_and_action_layout():
    nvec = squad_action_space_nvec()
    assert len(nvec) == SQUAD_ACTION_DIM
    assert int(sum(nvec)) == SQUAD_FLAT_ACTION_DIM
    assert nvec[SQUAD_ACTION_COMMAND_IDX] == CMD_DIM
    assert nvec[SQUAD_ACTION_TARGET_IDX] == SQUAD_TARGET_SLOTS
    assert nvec[SQUAD_ACTION_SPATIAL_IDX] == SQUAD_SPATIAL_CELLS


def test_target_slot_round_trip_identifies_allies_enemies_and_null():
    assert target_slot_kind(ally_target_slot(0)) == CMD_MODE_ALLY_TARGET
    assert target_slot_local_index(ally_target_slot(5)) == 5
    assert target_slot_kind(enemy_target_slot(0)) == CMD_MODE_ENEMY_TARGET
    assert target_slot_local_index(enemy_target_slot(7)) == 7
    assert target_slot_kind(SQUAD_NULL_TARGET_SLOT) == CMD_MODE_NONE


def test_target_tag_from_slot_uses_stable_slot_ordering():
    ally_tags = [10, 20, 30]
    enemy_tags = [40, 50]
    assert target_tag_from_slot(ally_target_slot(1), ally_tags, enemy_tags) == 20
    assert target_tag_from_slot(enemy_target_slot(1), ally_tags, enemy_tags) == 50
    assert target_tag_from_slot(SQUAD_NULL_TARGET_SLOT, ally_tags, enemy_tags) is None
    # Padded slot with no live unit behind it resolves to no order.
    assert target_tag_from_slot(ally_target_slot(9), ally_tags, enemy_tags) is None


def test_command_target_modes_table():
    from src.agent.squad_constants import (
        CMD_MODE_AOE,
        CMD_MODE_ATTACK_LOCAL,
        CMD_MODE_CLOAK,
        CMD_MODE_UNCLOAK,
        CMD_MODE_UNLOAD,
    )

    modes = command_target_modes(
        [
            "move_global",
            "attack_local",
            "attack_unit",
            "heal_unit",
            "load_unit",
            "aoe_local",
            "unload_local",
        ],
        ["cloak_burrow", "uncloak_unburrow"],
    )
    assert modes[0] == CMD_MODE_SPATIAL
    assert modes[1] == CMD_MODE_ATTACK_LOCAL
    assert modes[2] == CMD_MODE_ENEMY_TARGET
    assert modes[3] == CMD_MODE_ALLY_TARGET
    assert modes[4] == CMD_MODE_LOAD_TARGET
    assert modes[5] == CMD_MODE_AOE
    assert modes[6] == CMD_MODE_UNLOAD
    assert modes[7] == CMD_MODE_CLOAK
    assert modes[8] == CMD_MODE_UNCLOAK
    assert modes[9] == CMD_MODE_NONE  # padded _unused slot


def test_selected_tags_from_indices():
    tags = [100, 200, 300]
    assert selected_tags_from_indices(tags, [0, 2]) == [100, 300]

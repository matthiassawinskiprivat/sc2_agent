"""Tests for squad pointer selection resolver."""

import numpy as np

from src.agent.squad_selection import (
    SquadSelectionConfig,
    resolve_squad_selection,
    decode_squad_action,
    selected_tags_from_indices,
    resolve_squad_selection_from_binary,
)
from src.agent.squad_constants import MAX_SQUAD_UNITS, SQUAD_SPATIAL_CELLS
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


def test_decode_squad_action_layout():
    action = np.zeros(MAX_SQUAD_UNITS + 2, dtype=np.int64)
    action[0] = 1
    action[MAX_SQUAD_UNITS] = 3
    action[MAX_SQUAD_UNITS + 1] = 42
    mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    mask[0] = 1.0
    selection, cmd, spatial = decode_squad_action(action, mask)
    assert 0 in selection.selected_indices
    assert cmd == 3
    assert spatial == 42


def test_selected_tags_from_indices():
    tags = [100, 200, 300]
    assert selected_tags_from_indices(tags, [0, 2]) == [100, 300]

"""Tests for debug UI spatial grid → policy action helpers."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from sc2.position import Point2

from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
from src.ui.spatial_debug_control import (
    build_universal_spatial_action,
    command_head_index,
    grid_from_spatial_idx,
    parse_grid_field,
    spatial_idx_from_grid,
)


def test_spatial_idx_from_grid_corners():
    assert spatial_idx_from_grid(0, 0) == 0
    assert spatial_idx_from_grid(31, 31) == 1023
    assert grid_from_spatial_idx(1023) == (31, 31)


def test_parse_grid_field_clamps():
    assert parse_grid_field("0") == 0
    assert parse_grid_field("31") == 31
    assert parse_grid_field("999") == 31
    assert parse_grid_field("-5") == 0


def test_build_universal_spatial_action_uses_manifest():
    conf = {
        "action_heads": ["move_global", "move_local"],
        "action_head_manifest": ["move_global", "move_local"],
    }
    action = build_universal_spatial_action("move_local", 5, 7, conf)
    assert action.tolist() == [command_head_index(conf, "move_local"), spatial_idx_from_grid(5, 7)]


def test_move_global_corners_decode_to_playable_edges():
    """Grid (0,0) / (31,31) should land near playable NW / SE on a letterboxed ladder map."""
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    gi = MagicMock()
    gi.map_size.width = 512
    gi.map_size.height = 512
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = 512
    gi.playable_area.height = 512
    gi.pathing_grid = MagicMock(data_numpy=np.ones((512, 512), dtype=np.uint8))
    agent.last_game_info = gi

    focal = MagicMock()
    focal.position = Point2((256.0, 256.0))

    nw_x, nw_y = agent.convert_spatial_action_to_world(
        "move_global", spatial_idx_from_grid(0, 0), active_unit=focal
    )
    se_x, se_y = agent.convert_spatial_action_to_world(
        "move_global", spatial_idx_from_grid(31, 31), active_unit=focal
    )

    assert nw_x < se_x
    assert nw_y > se_y
    assert 0.0 <= nw_x <= 512.0
    assert 0.0 <= se_x <= 512.0


def test_move_local_grid_maps_to_tactical_offset():
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    gi = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = 128
    gi.playable_area.height = 128
    gi.map_size.width = 128
    gi.map_size.height = 128
    agent.last_game_info = gi

    focal = MagicMock()
    focal.position = Point2((64.0, 64.0))
    idx = spatial_idx_from_grid(16, 15)
    wx, wy = agent.convert_spatial_action_to_world("move_local", idx, active_unit=focal)
    assert wx == pytest.approx(64.0)
    assert wy == pytest.approx(64.0)

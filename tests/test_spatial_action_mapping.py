import numpy as np
from unittest.mock import MagicMock

from src.agent.unit_control_agent import UnitControlAgent
from src.agent.universal_unit_control_agent import UniversalUnitControlAgent


def _mock_game_info(map_w: int, map_h: int, area_x: int, area_y: int, area_w: int, area_h: int):
    gi = MagicMock()
    gi.map_size.width = map_w
    gi.map_size.height = map_h
    gi.playable_area.x = area_x
    gi.playable_area.y = area_y
    gi.playable_area.width = area_w
    gi.playable_area.height = area_h
    return gi


def test_global_decode_clamped_to_playable_area():
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = _mock_game_info(map_w=200, map_h=180, area_x=10, area_y=20, area_w=128, area_h=96)

    # Sample a few edge indices; they should always land inside playable_area
    indices = [0, 31, 32 * 31, 32 * 32 - 1, 511, 512]
    for idx in indices:
        x, y = agent.convert_spatial_action_to_world(idx)
        assert 10.0 <= x < 10.0 + 128.0
        assert 20.0 <= y < 20.0 + 96.0


def test_global_roundtrip_is_stable_within_cell():
    agent = UnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = _mock_game_info(map_w=200, map_h=180, area_x=7, area_y=9, area_w=150, area_h=150)

    rng = np.random.default_rng(0)
    for _ in range(100):
        x = float(rng.uniform(7.0, 7.0 + 150.0 - 1e-3))
        y = float(rng.uniform(9.0, 9.0 + 150.0 - 1e-3))
        pos = MagicMock()
        pos.x = x
        pos.y = y

        idx = agent.get_grid_index_at_world_pos(pos)
        x2, y2 = agent.convert_spatial_action_to_world(idx)

        # Returned point should always be inside playable area.
        assert 7.0 <= x2 < 7.0 + 150.0
        assert 9.0 <= y2 < 9.0 + 150.0


def test_local_decode_clamped_to_playable_area():
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = _mock_game_info(map_w=200, map_h=180, area_x=10, area_y=20, area_w=128, area_h=96)

    active_unit = MagicMock()
    active_unit.position.x = 10.0  # at min corner of playable area
    active_unit.position.y = 20.0

    # Extreme local offsets should clamp back into playable.
    indices = [0, 32 * 32 - 1]
    for idx in indices:
        x, y = agent.convert_spatial_action_to_world("move_local", idx, active_unit=active_unit)
        assert 10.0 <= x < 10.0 + 128.0
        assert 20.0 <= y < 20.0 + 96.0


# --- get_action_mask shape tests ---

def test_universal_get_action_mask_no_heads_returns_spatial_only():
    """head_names=None → flat mask of size spatial_resolution^2 (1024)."""
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    mask = agent.get_action_mask(head_names=None)
    assert mask.shape == (1024,)
    assert mask.dtype == bool
    assert mask.all()


def test_universal_get_action_mask_with_two_heads():
    """head_names with 2 entries → mask of size 2 + 1024 = 1026."""
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    mask = agent.get_action_mask(head_names=["move_global", "attack_global"])
    assert mask.shape == (1026,)
    assert mask.all()


def test_universal_get_action_mask_with_three_heads():
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    mask = agent.get_action_mask(head_names=["move_global", "move_local", "attack_global"])
    assert mask.shape == (1027,)
    assert mask[:3].all()
    assert mask[3:].all()


"""
Regression test: opponent (player 2) spatial actions must be perspective-un-flipped
before being converted to world coordinates.

The policy sees a 180°-rotated observation for player 2, so it emits a spatial
index in *flipped* space.  apply_action must invert that flip before calling
convert_spatial_action_to_world so the unit moves to the correct location.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_bot(player_id: int = 1) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(
        player_id=player_id,
        training_conf={
            "agent_architecture": "universal",
            "action_heads": ["move", "attack_local"],
            "categorical_heads": [],
            "grid_size": 128,
        },
    )
    # Minimal SC2 state mocks
    bot.agent_system = MagicMock()
    bot.do = MagicMock()
    return bot


def _make_unit(owner: int) -> MagicMock:
    unit = MagicMock()
    unit.owner_id = owner  # sc2 Python API uses owner_id, not owner
    unit.position = MagicMock()
    unit.attack = MagicMock(return_value=MagicMock())
    unit.move = MagicMock(return_value=MagicMock())
    return unit


def _expected_flip(spatial_idx: int, grid_size: int = 32) -> int:
    row, col = divmod(spatial_idx, grid_size)
    return (grid_size - 1 - row) * grid_size + (grid_size - 1 - col)


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
def test_opponent_unit_spatial_idx_is_unflipped(spatial_idx):
    """
    For a player-2 unit the spatial index must be un-flipped (180° rotation
    on a 32×32 grid) before convert_spatial_action_to_world is called.
    """
    bot = _make_bot(player_id=1)
    opp_unit = _make_unit(owner=2)

    # all_units.find_by_tag returns the opponent unit
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = opp_unit

    # Track what spatial_idx is actually passed to convert_spatial_action_to_world
    captured = {}

    def fake_convert(head_name, s_idx, active_unit=None):
        captured["spatial_idx"] = s_idx
        return (64.0, 64.0)

    bot.agent_system.unit_agents = {
        "active_agent": MagicMock(convert_spatial_action_to_world=fake_convert)
    }

    action = np.array([0, spatial_idx])  # head_idx=0 ("move"), spatial_idx
    asyncio.run(bot.apply_action(unit_tag=99, action=action))

    assert "spatial_idx" in captured, "convert_spatial_action_to_world was not called"
    expected = _expected_flip(spatial_idx)
    assert captured["spatial_idx"] == expected, (
        f"spatial_idx={spatial_idx}: expected flipped={expected}, "
        f"got {captured['spatial_idx']}"
    )


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
def test_own_unit_spatial_idx_is_not_flipped(spatial_idx):
    """
    For a player-1 unit the spatial index must be passed through unchanged.
    """
    bot = _make_bot(player_id=1)
    own_unit = _make_unit(owner=1)

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = own_unit

    captured = {}

    def fake_convert(head_name, s_idx, active_unit=None):
        captured["spatial_idx"] = s_idx
        return (64.0, 64.0)

    bot.agent_system.unit_agents = {
        "active_agent": MagicMock(convert_spatial_action_to_world=fake_convert)
    }

    action = np.array([0, spatial_idx])
    asyncio.run(bot.apply_action(unit_tag=42, action=action))

    assert "spatial_idx" in captured, "convert_spatial_action_to_world was not called"
    assert captured["spatial_idx"] == spatial_idx, (
        f"Own unit spatial_idx should be unchanged: expected {spatial_idx}, "
        f"got {captured['spatial_idx']}"
    )


def test_flip_is_its_own_inverse():
    """
    Applying the 180° flip twice must yield the original index.
    """
    for idx in [0, 1, 31, 32, 100, 512, 993, 1023]:
        once = _expected_flip(idx)
        twice = _expected_flip(once)
        assert twice == idx, f"Double-flip of {idx} gave {twice}, expected {idx}"


def test_padded_command_head_is_no_op():
    """
    A command head index beyond the configured list must be a no-op (no SC2 order).
    """
    bot = _make_bot(player_id=1)
    opp_unit = _make_unit(owner=2)

    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = opp_unit

    captured = {}
    bot.agent_system.unit_agents = {
        "active_agent": MagicMock(
            convert_spatial_action_to_world=lambda *a, **k: captured.update({"called": True}) or (0.0, 0.0)
        )
    }

    # head_idx=5 is beyond the 2 configured heads → padded slot → no-op
    action = np.array([5, 100])
    asyncio.run(bot.apply_action(unit_tag=99, action=action))

    assert "called" not in captured, "convert_spatial_action_to_world should not be called for padded head"

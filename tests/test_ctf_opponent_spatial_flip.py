"""
Spatial actions are applied in world coordinates for all SC2 owners.

P1 and P2 use the same observation/action frame as legacy P1-only training:
no 180° map rotation and no spatial index un-flip on apply.
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock

from sc2.ids.unit_typeid import UnitTypeId

from src.agent.ctf_bot import CaptureTheFlagBot


def _make_bot(player_id: int = 1, action_heads=None) -> CaptureTheFlagBot:
    bot = CaptureTheFlagBot(
        player_id=player_id,
        training_conf={
            "agent_architecture": "universal",
            "action_heads": list(action_heads or ["move_global", "attack_local"]),
            "categorical_heads": [],
            "grid_size": 128,
        },
    )
    bot.agent_system = MagicMock()
    bot.do = MagicMock()
    return bot


def _make_unit(owner: int) -> MagicMock:
    unit = MagicMock()
    unit.owner_id = owner
    unit.type_id = UnitTypeId.MARINE
    unit.position = MagicMock()
    unit.attack = MagicMock(return_value=MagicMock())
    unit.move = MagicMock(return_value=MagicMock())
    return unit


def _capture_convert(bot):
    captured = {}

    def fake_convert(head_name, s_idx, active_unit=None):
        captured["spatial_idx"] = s_idx
        captured["head_name"] = head_name
        return (64.0, 64.0)

    bot.agent_system.unit_agents = {
        "active_agent": MagicMock(convert_spatial_action_to_world=fake_convert)
    }
    return captured


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
@pytest.mark.parametrize("owner", [1, 2])
def test_apply_action_local_spatial_idx_is_not_flipped(spatial_idx, owner):
    bot = _make_bot(action_heads=["attack_local"])
    unit = _make_unit(owner=owner)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    captured = _capture_convert(bot)

    asyncio.run(bot.apply_action(unit_tag=99, action=np.array([0, spatial_idx])))

    assert captured.get("spatial_idx") == spatial_idx


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
def test_apply_action_global_spatial_idx_unchanged_for_p2(spatial_idx):
    bot = _make_bot(action_heads=["move_global"])
    unit = _make_unit(owner=2)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    captured = _capture_convert(bot)

    asyncio.run(bot.apply_action(unit_tag=99, action=np.array([0, spatial_idx])))

    assert captured.get("spatial_idx") == spatial_idx


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
def test_apply_action_global_spatial_idx_unchanged_for_p1(spatial_idx):
    bot = _make_bot(action_heads=["move_global"])
    unit = _make_unit(owner=1)
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    captured = _capture_convert(bot)

    asyncio.run(bot.apply_action(unit_tag=99, action=np.array([0, spatial_idx])))

    assert captured.get("spatial_idx") == spatial_idx


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
@pytest.mark.parametrize("perspective", [1, 2])
def test_apply_squad_action_local_spatial_idx_is_not_flipped(spatial_idx, perspective):
    bot = _make_bot(action_heads=["attack_local"])
    unit = _make_unit(owner=perspective)
    unit.tag = 99
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    captured = _capture_convert(bot)

    asyncio.run(
        bot.apply_squad_action(
            [99],
            anchor_tag=99,
            command_idx=0,
            spatial_idx=spatial_idx,
            perspective_player_id=perspective,
        )
    )

    assert captured.get("spatial_idx") == spatial_idx


@pytest.mark.parametrize("spatial_idx", [0, 1, 100, 512, 1023])
def test_apply_squad_action_global_spatial_idx_unchanged_for_p2(spatial_idx):
    bot = _make_bot(action_heads=["move_global"])
    unit = _make_unit(owner=2)
    unit.tag = 99
    bot.all_units = MagicMock()
    bot.all_units.find_by_tag.return_value = unit
    captured = _capture_convert(bot)

    asyncio.run(
        bot.apply_squad_action(
            [99],
            anchor_tag=99,
            command_idx=0,
            spatial_idx=spatial_idx,
            perspective_player_id=2,
        )
    )

    assert captured.get("spatial_idx") == spatial_idx


def test_padded_command_head_is_no_op():
    """A command head index beyond the configured list must be a no-op."""
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

    action = np.array([5, 100])
    asyncio.run(bot.apply_action(unit_tag=99, action=action))

    assert "called" not in captured

"""Verify command_objective rendering for each environment's command_target contract."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.transformers.observation_transformer import ObservationTransformer
from tests.obs_transformer_test_utils import get_layer, transform_map
from tests.test_observation_transformer import _make_game_info, create_mock_unit


def _game_info_128():
    gi = _make_game_info(128, 128, with_grids=True)
    gi.playable_area = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = 128
    gi.playable_area.height = 128
    return gi


def _focal_unit():
    return create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)), radius=0.5)


def _objective_sums(raw_obs, active_unit=None):
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["command_objective"],
        "active_tactical_layers": ["command_objective"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    global_sum = float(
        get_layer(
            transform_map(transformer, raw_obs),
            "command_objective",
            ["command_objective"],
        ).sum()
    )
    tactical_sum = 0.0
    if active_unit is not None:
        tactical_sum = float(transformer.transform_tactical(raw_obs, active_unit)[0].sum())
    return global_sum, tactical_sum


def _game_info_ladder_playable(width=512, height=512):
    gi = _make_game_info(width, height, with_grids=True)
    gi.playable_area = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = width
    gi.playable_area.height = height
    gi.map_name = "ladder_playable"
    return gi


def _global_objective_area(raw_obs) -> int:
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["command_objective"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    layer = get_layer(
        transform_map(transformer, raw_obs),
        "command_objective",
        ["command_objective"],
    )
    return int(np.count_nonzero(layer > 0))


@pytest.fixture
def base_raw_obs():
    active = _focal_unit()
    return {
        "units": [active],
        "enemy_units": [],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _game_info_128(),
        "game_data": MagicMock(),
        "selected_units": [active],
        "active": active,
    }


def test_level0_point2_beacon_renders_global_and_tactical(base_raw_obs):
    """Level0/SC2Bot: command_target = beacon.position (Point2)."""
    raw = {**base_raw_obs, "command_target": Point2((52.0, 52.0))}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g > 0
    assert t > 0


def test_level3_unit_beacon_renders_strong_global_marker(base_raw_obs):
    """Level3Bot: command_target = [assigned_beacon Unit] (not bare Point2)."""
    beacon = create_mock_unit(
        UnitTypeId.BEACON_TERRANSMALL, Point2((52.0, 52.0)), radius=1.5
    )
    raw = {**base_raw_obs, "command_target": [beacon]}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g > 0
    assert t > 0


def test_level7_point2_enemy_renders_when_near(base_raw_obs):
    """Level7Bot: command_target = live enemy.position (Point2)."""
    raw = {**base_raw_obs, "command_target": Point2((54.0, 48.0))}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g > 0
    assert t > 0


def test_level_drop_fog_persisted_point2_renders_on_global(base_raw_obs):
    """LevelDropBot: command_target = _objective_pos (Point2) even when structure not visible."""
    raw = {**base_raw_obs, "command_target": Point2((100.0, 100.0))}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g > 0
    # Tactical fovea is 32 gu — far objective is intentionally skipped.
    assert t == 0.0


def test_ctf_spawn_time_point2_renders_on_global(base_raw_obs):
    """CTF (single-env contract): unit_target_map Point2 always paints global objective."""
    raw = {**base_raw_obs, "command_target": Point2((100.0, 100.0))}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g > 0
    assert t == 0.0


def test_ctf_missing_target_renders_nothing(base_raw_obs):
    """CaptureTheFlagEnv attacker fallback when pylon unseen: command_target is None."""
    raw = {**base_raw_obs, "command_target": None}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g == 0.0
    assert t == 0.0


def test_ctf_defender_near_pylon_renders_both_views(base_raw_obs):
    """CTF defenders: own pylon position as Point2 near focal unit."""
    raw = {**base_raw_obs, "command_target": Point2((48.0, 48.0))}
    g, t = _objective_sums(raw, base_raw_obs["active"])
    assert g > 0
    assert t > 0


@pytest.mark.parametrize(
    "contract_name,target",
    [
        ("level7_enemy_point2", Point2((256.0, 256.0))),
        ("ctf_pylon_point2", Point2((256.0, 256.0))),
    ],
)
def test_attack_and_ctf_objective_radius_constant_on_ladder(contract_name, target):
    """
    Level7Bot and CTFBot both pass command_target as Point2 (enemy / pylon position).
    Ladder-sized playables must keep the same global objective footprint as 128×128.
    """
    del contract_name  # documentation only
    active = _focal_unit()
    raw_128 = {
        "units": [active],
        "enemy_units": [],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _game_info_128(),
        "game_data": MagicMock(),
        "selected_units": [active],
        "command_target": Point2((64.0, 64.0)),
    }
    raw_ladder = {**raw_128, "game_info": _game_info_ladder_playable(), "command_target": target}
    area_128 = _global_objective_area(raw_128)
    area_ladder = _global_objective_area(raw_ladder)
    assert area_128 >= 20
    assert area_ladder == area_128

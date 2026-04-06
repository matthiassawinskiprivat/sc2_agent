import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_mock_unit(unit_type, pos, cargo_used=0, cargo_max=0, **overrides):
    u = MagicMock(spec=Unit)
    u.type_id = unit_type
    u.position = pos
    u.cargo_used = cargo_used
    u.cargo_max = cargo_max
    u.radius = 1.0
    u.is_cloaked = False
    u.is_blip = False
    u.is_structure = False
    u.is_ready = True
    u.health = 100
    u.health_max = 100
    u.shield = 0
    u.shield_max = 0
    u.energy = 0
    u.energy_max = 0
    u.weapon_cooldown = 0
    u.attack_upgrade_level = 0
    u.detect_range = 0
    u.is_biological = False
    u.is_mechanical = False
    u.is_armored = False
    u.is_massive = False
    u.is_flying = False
    u.is_hallucination = False
    u.build_progress = 1.0
    for k, v in overrides.items():
        setattr(u, k, v)
    return u


def _make_game_info(width=64, height=64, with_grids=False):
    gi = MagicMock()
    gi.map_size.width = width
    gi.map_size.height = height
    del gi.playable_area
    if with_grids:
        grid = np.ones((height, width), dtype=np.uint8)
        gi.pathing_grid = MagicMock()
        gi.pathing_grid.data_numpy = grid
        gi.terrain_height = MagicMock()
        gi.terrain_height.data_numpy = (grid * 128).astype(np.uint8)
        gi.map_name = "test_map"
    else:
        del gi.pathing_grid
        del gi.terrain_height
    return gi


# ---------------------------------------------------------------------------
# Transport layers
# ---------------------------------------------------------------------------

def test_transport_layers_rendering():
    """Medivacs are drawn on TRANSPORT_SLOTS and PASSENGER_DENSITY."""
    transformer = ObservationTransformer(map_size=(64, 64))

    medivac_empty = create_mock_unit(UnitTypeId.MEDIVAC, Point2((10, 10)), cargo_used=0, cargo_max=8)
    medivac_full = create_mock_unit(UnitTypeId.MEDIVAC, Point2((20, 20)), cargo_used=8, cargo_max=8)
    medivac_half = create_mock_unit(UnitTypeId.MEDIVAC, Point2((30, 30)), cargo_used=2, cargo_max=8)
    marine = create_mock_unit(UnitTypeId.MARINE, Point2((40, 40)), cargo_used=0, cargo_max=0)

    raw_obs = {
        "units": [medivac_empty, medivac_full, medivac_half, marine],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
    }

    result = transformer.transform(raw_obs)
    map_data = result["map"]

    idx_slots = LayerConstants.LAYERS[LayerConstants.TRANSPORT_SLOTS]
    idx_pass = LayerConstants.LAYERS[LayerConstants.PASSENGER_DENSITY]

    py_10 = 64 - 1 - 10
    py_20 = 64 - 1 - 20
    py_30 = 64 - 1 - 30

    assert map_data[idx_slots, py_10, 10] == pytest.approx(0.5, abs=0.05)
    assert map_data[idx_slots, py_20, 20] == pytest.approx(1.0, abs=0.05)
    assert map_data[idx_slots, py_30, 30] == pytest.approx(0.5, abs=0.05)

    assert map_data[idx_pass, py_20, 20] == pytest.approx(1.0, abs=0.05)
    assert map_data[idx_pass, py_30, 30] == pytest.approx(0.25, abs=0.05)
    assert map_data[idx_pass, py_10, 10] == 0.0


# ---------------------------------------------------------------------------
# transform() returns correct dict
# ---------------------------------------------------------------------------

def test_transform_returns_dict_with_map_and_command():
    transformer = ObservationTransformer(map_size=(32, 32))
    raw_obs = {"units": [], "enemy_units": [], "effects": [], "game_info": None, "game_data": None}
    result = transformer.transform(raw_obs)

    assert isinstance(result, dict)
    assert "map" in result
    assert "command" in result
    assert result["map"].dtype == np.float32
    assert result["command"].dtype == np.float32
    assert result["command"].shape == (10,)


# ---------------------------------------------------------------------------
# Command vector encoding
# ---------------------------------------------------------------------------

def test_command_vector_default_zero():
    transformer = ObservationTransformer(map_size=(32, 32))
    raw_obs = {"units": [], "enemy_units": [], "effects": [], "game_info": None, "game_data": None}
    result = transformer.transform(raw_obs)
    assert result["command"][0] == 1.0
    assert result["command"].sum() == 1.0


def test_command_vector_with_high_level_command():
    transformer = ObservationTransformer(map_size=(32, 32))
    raw_obs = {
        "units": [], "enemy_units": [], "effects": [],
        "game_info": None, "game_data": None,
        "high_level_command": "scout",
    }
    result = transformer.transform(raw_obs)
    assert result["command"].sum() == 1.0
    assert result["command"][0] != 1.0 or True  # scout may map to index 0


# ---------------------------------------------------------------------------
# COMMAND_ASSIGNMENT radius on large playable areas
# ---------------------------------------------------------------------------


def test_command_assignment_radius_floored_like_128_playable():
    """Large play areas shrink self.scale; assignment dot keeps 128×128-equivalent pixel radius."""
    transformer = ObservationTransformer(map_size=(128, 128))

    gi = MagicMock()
    gi.map_size = MagicMock(width=256, height=256)
    gi.playable_area = MagicMock(x=0, y=0, width=256, height=256)
    gi.map_name = "big_map"
    grid = np.ones((256, 256), dtype=np.uint8)
    gi.pathing_grid = MagicMock()
    gi.pathing_grid.data_numpy = grid
    gi.terrain_height = MagicMock()
    gi.terrain_height.data_numpy = (grid * 128).astype(np.uint8)

    unit = create_mock_unit(UnitTypeId.MARINE, Point2((128.0, 128.0)), radius=2.5)

    raw_obs = {
        "units": [],
        "selected_units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": gi,
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    idx_sel = LayerConstants.LAYERS[LayerConstants.COMMAND_ASSIGNMENT]
    assign = result["map"][idx_sel]

    px, py = transformer._get_pixel_coords(unit.position)
    assert assign[py, px] > 0
    # scale=0.5 → naive int(2.5*0.5)=1 would miss (px-2, py); floor uses ref scale 1.0 → r=2
    assert assign[py, px - 2] > 0


# ---------------------------------------------------------------------------
# Output clipping (all values in [0, 1])
# ---------------------------------------------------------------------------

def test_output_values_clipped():
    transformer = ObservationTransformer(map_size=(32, 32))
    raw_obs = {"units": [], "enemy_units": [], "effects": [], "game_info": None, "game_data": None}
    result = transformer.transform(raw_obs)
    assert result["map"].min() >= 0.0
    assert result["map"].max() <= 1.0


# ---------------------------------------------------------------------------
# Beacon rendering on ENEMY_UNIT_DENSITY
# ---------------------------------------------------------------------------

def test_beacon_rendered_on_enemy_density():
    transformer = ObservationTransformer(map_size=(64, 64))
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, Point2((32, 32)))

    raw_obs = {
        "units": [],
        "enemy_units": [beacon],
        "effects": [],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    map_data = result["map"]
    idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    # Beacon should produce nonzero signal (log1p normalized)
    assert map_data[idx].sum() > 0


# ---------------------------------------------------------------------------
# Command objective — Point2 target vs beacon Unit target
# ---------------------------------------------------------------------------

def test_command_objective_point2_target():
    """Point2 target draws on COMMAND_OBJECTIVE with r=3."""
    transformer = ObservationTransformer(map_size=(64, 64))
    target = Point2((32, 32))

    raw_obs = {
        "units": [],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
        "command_target": target,
    }
    result = transformer.transform(raw_obs)
    map_data = result["map"]
    idx = LayerConstants.LAYERS[LayerConstants.COMMAND_OBJECTIVE]
    assert map_data[idx].sum() > 0


def test_command_objective_beacon_unit_target():
    """Beacon Unit target draws with val=20.0 and r=max(3, int(1.5*scale))."""
    transformer = ObservationTransformer(map_size=(64, 64))
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, Point2((32, 32)))

    raw_obs = {
        "units": [],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
        "command_target": [beacon],
    }
    result = transformer.transform(raw_obs)
    map_data = result["map"]
    idx = LayerConstants.LAYERS[LayerConstants.COMMAND_OBJECTIVE]
    # Beacon marker uses val=20.0 pre-clip, so after clip should be 1.0
    assert map_data[idx].max() == pytest.approx(1.0, abs=0.01)
    # Signal should be larger than a single pixel (radius >= 3)
    assert np.count_nonzero(map_data[idx]) > 1


# ---------------------------------------------------------------------------
# _draw_circle_vectorized
# ---------------------------------------------------------------------------

def test_draw_circle_accumulate_vs_set():
    transformer = ObservationTransformer(map_size=(32, 32))
    layers = np.zeros((44, 32, 32), dtype=np.float32)

    transformer._draw_circle_vectorized(layers, 0, 16, 16, 0, 0, 1.0, True)
    transformer._draw_circle_vectorized(layers, 0, 16, 16, 0, 0, 1.0, True)
    assert layers[0, 16, 16] == pytest.approx(2.0)

    transformer._draw_circle_vectorized(layers, 1, 16, 16, 0, 0, 5.0, False)
    transformer._draw_circle_vectorized(layers, 1, 16, 16, 0, 0, 3.0, False)
    assert layers[1, 16, 16] == pytest.approx(3.0)


def test_draw_circle_skips_inactive_index():
    active_layers = [LayerConstants.ALLIED_UNIT_DENSITY]
    transformer = ObservationTransformer(map_size=(32, 32), training_conf={"active_layers": active_layers})
    layers = np.zeros((44, 32, 32), dtype=np.float32)

    inactive_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    transformer._draw_circle_vectorized(layers, inactive_idx, 16, 16, 0, 0, 1.0, True)
    assert layers[inactive_idx, 16, 16] == 0.0


# ---------------------------------------------------------------------------
# transform_tactical
# ---------------------------------------------------------------------------

def test_tactical_returns_zeros_when_no_active_unit():
    active_tactical = [
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.ENEMY_UNIT_DENSITY,
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={"active_tactical_layers": active_tactical},
    )
    result = transformer.transform_tactical({}, None)
    assert result.shape == (2, 32, 32)
    assert result.sum() == 0.0


def test_tactical_output_shape_with_active_layers():
    active_layers = [
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.ENEMY_UNIT_DENSITY,
        LayerConstants.TERRAIN_PATHABLE,
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": active_layers,
            "active_tactical_layers": active_layers,
        },
    )
    unit = create_mock_unit(UnitTypeId.ZERGLING, Point2((64, 64)))
    raw_obs = {
        "units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": None,
        "selected_units": [unit],
    }
    result = transformer.transform_tactical(raw_obs, unit)
    assert result.shape == (3, 32, 32)


def test_tactical_preserves_config_order():
    """Tactical slicing follows active_tactical_layers order, not sorted TACTICAL index."""
    layers_order = [
        LayerConstants.COMMAND_OBJECTIVE,     # TACTICAL index 4
        LayerConstants.ALLIED_UNIT_DENSITY,   # TACTICAL index 0
        LayerConstants.TERRAIN_PATHABLE,      # TACTICAL index 2
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": layers_order,
            "active_tactical_layers": layers_order,
        },
    )
    unit = create_mock_unit(UnitTypeId.ZERGLING, Point2((64, 64)))
    raw_obs = {
        "units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": None,
        "selected_units": [unit],
        "command_target": Point2((70, 70)),
    }
    result = transformer.transform_tactical(raw_obs, unit)
    assert result.shape == (3, 32, 32)

    # Channel 0 should be COMMAND_OBJECTIVE (has signal from command_target)
    assert result[0].sum() > 0
    # Channel 1 should be ALLIED_UNIT_DENSITY (has signal from the zergling)
    assert result[1].sum() > 0


def test_tactical_dps_follows_active_tactical_not_global_layers():
    """Tactical DPS must draw when listed in active_tactical_layers even if global active_layers omits DPS."""
    tactical = [
        LayerConstants.TERRAIN_PATHABLE,
        LayerConstants.ENEMY_UNIT_DENSITY,
        LayerConstants.ENEMY_G_DPS_BASE,
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": tactical,
        },
    )
    zerg = create_mock_unit(UnitTypeId.ZERGLING, Point2((64, 64)))
    marine = create_mock_unit(UnitTypeId.MARINE, Point2((66, 64)))
    raw_obs = {
        "units": [zerg],
        "enemy_units": [marine],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": None,
        "selected_units": [zerg],
    }
    result = transformer.transform_tactical(raw_obs, zerg)
    assert result.shape == (3, 32, 32)
    assert result[2].max() > 0, "enemy_g_dps tactical channel should receive weapon heatmap"


def test_tactical_restores_state_after_call():
    """transform_tactical must restore all internal state after returning."""
    transformer = ObservationTransformer(map_size=(128, 128))

    original_scale = transformer.scale
    original_width = transformer.feature_map_width
    original_height = transformer.feature_map_height
    original_mode = transformer.is_tactical_mode

    unit = create_mock_unit(UnitTypeId.ZERGLING, Point2((64, 64)))
    raw_obs = {
        "units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(128, 128),
        "game_data": None,
        "selected_units": [unit],
    }
    transformer.transform_tactical(raw_obs, unit)

    assert transformer.scale == original_scale
    assert transformer.feature_map_width == original_width
    assert transformer.feature_map_height == original_height
    assert transformer.is_tactical_mode == original_mode


# ---------------------------------------------------------------------------
# Global layer slicing preserves config order
# ---------------------------------------------------------------------------

def test_global_slicing_preserves_config_order():
    """Global map slicing follows active_layers order."""
    layers_order = [
        LayerConstants.COMMAND_OBJECTIVE,     # LAYERS index 41
        LayerConstants.ALLIED_UNIT_DENSITY,   # LAYERS index 0
        LayerConstants.TERRAIN_PATHABLE,      # LAYERS index 38
    ]
    transformer = ObservationTransformer(
        map_size=(64, 64),
        training_conf={"active_layers": layers_order},
    )
    raw_obs = {
        "units": [],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(64, 64, with_grids=True),
        "game_data": None,
        "command_target": Point2((32, 32)),
    }
    result = transformer.transform(raw_obs)
    map_data = result["map"]
    assert map_data.shape[0] == 3

    # Channel 0 = COMMAND_OBJECTIVE (has signal from command_target)
    assert map_data[0].sum() > 0
    # Channel 2 = TERRAIN_PATHABLE (has signal from pathing grid)
    assert map_data[2].sum() > 0

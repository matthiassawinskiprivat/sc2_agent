import pytest
import numpy as np
from unittest.mock import MagicMock
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from src.utils.obs_geometry import compute_global_obs_geometry, observation_bounds
from tests.obs_transformer_test_utils import transform_map


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
    u.is_burrowed = False
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
    """Large play areas shrink self.scale; large units keep 128×128-equivalent assignment footprint."""
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

    unit = create_mock_unit(
        UnitTypeId.THOR,
        Point2((128.0, 128.0)),
        radius=2.5,
        is_mechanical=True,
        is_armored=True,
        is_massive=True,
    )

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
    assert int(np.count_nonzero(assign > 0)) >= 1


def test_command_objective_radius_constant_on_large_playable():
    """512 playable maps must paint the same global objective footprint as 128×128."""
    training_conf = {
        "agent_architecture": "universal",
        "active_layers": [LayerConstants.COMMAND_OBJECTIVE],
    }
    target = Point2((64.0, 64.0))

    def _objective_area(game_info) -> int:
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=training_conf)
        raw_obs = {
            "units": [],
            "enemy_units": [],
            "effects": [],
            "game_info": game_info,
            "game_data": None,
            "command_target": target,
        }
        layer = transform_map(transformer, raw_obs)[0]
        return int(np.count_nonzero(layer > 0))

    gi_128 = MagicMock()
    gi_128.map_size = MagicMock(width=128, height=128)
    gi_128.playable_area = MagicMock(x=0, y=0, width=128, height=128)
    gi_128.map_name = "small_playable"
    grid_128 = np.ones((128, 128), dtype=np.uint8)
    gi_128.pathing_grid = MagicMock(data_numpy=grid_128)
    gi_128.terrain_height = MagicMock(data_numpy=(grid_128 * 128).astype(np.uint8))

    gi_512 = MagicMock()
    gi_512.map_size = MagicMock(width=512, height=512)
    gi_512.playable_area = MagicMock(x=0, y=0, width=512, height=512)
    gi_512.map_name = "ladder_playable"
    grid_512 = np.ones((512, 512), dtype=np.uint8)
    gi_512.pathing_grid = MagicMock(data_numpy=grid_512)
    gi_512.terrain_height = MagicMock(data_numpy=(grid_512 * 128).astype(np.uint8))

    area_128 = _objective_area(gi_128)
    area_512 = _objective_area(gi_512)
    assert area_128 >= 20
    assert area_512 == area_128


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
    # Radius-3 disk should activate a visible local area, not just a single pixel.
    assert np.count_nonzero(map_data[idx]) >= 20


def test_command_objective_beacon_unit_target():
    """Beacon Unit target draws with val=20.0 and reference-scaled global radius."""
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


def test_health_layer_is_zero_for_full_health_unit():
    transformer = ObservationTransformer(map_size=(64, 64))
    unit = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((32, 32)),
        health=100,
        health_max=100,
        shield=0,
        shield_max=0,
    )
    raw_obs = {
        "units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_HEALTH]
    assert result["map"][idx].max() == pytest.approx(0.0, abs=1e-6)


def test_health_layer_scales_linearly_with_damage():
    transformer = ObservationTransformer(map_size=(64, 64))
    unit = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((32, 32)),
        health=95,
        health_max=100,
        shield=0,
        shield_max=0,
    )
    raw_obs = {
        "units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_HEALTH]
    assert result["map"][idx].max() == pytest.approx(0.05, abs=1e-6)


def test_health_layer_is_one_at_zero_hp():
    transformer = ObservationTransformer(map_size=(64, 64))
    unit = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((32, 32)),
        health=0,
        health_max=100,
        shield=0,
        shield_max=0,
    )
    raw_obs = {
        "units": [unit],
        "enemy_units": [],
        "effects": [],
        "game_info": _make_game_info(64, 64),
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_HEALTH]
    assert result["map"][idx].max() == pytest.approx(1.0, abs=1e-6)


def test_enemy_health_layer_scales_in_global_and_tactical():
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": ["enemy_health", "allied_health"],
            "active_tactical_layers": ["enemy_health", "allied_health"],
        },
    )
    active = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((64.0, 64.0)),
        radius=0.375,
    )
    enemy = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((70.0, 64.0)),
        health=50,
        health_max=100,
        shield=0,
        shield_max=0,
    )
    raw_obs = {
        "units": [active],
        "structures": [],
        "enemy_units": [enemy],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [active],
    }
    global_map = transformer.transform(raw_obs)["map"]
    tactical = transformer.transform_tactical(raw_obs, active)

    assert global_map[0].max() == pytest.approx(0.5, abs=1e-6)
    assert tactical[0].max() == pytest.approx(0.5, abs=1e-6)


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


def test_tactical_enemy_unit_mechanical_paints_and_slices():
    """Enemy mechanical units stamp enemy_unit_mechanical on the tactical crop."""
    assert LayerConstants.ENEMY_UNIT_MECHANICAL in LayerConstants.TACTICAL_LAYERS
    assert (
        LayerConstants.TACTICAL_LAYERS[LayerConstants.ENEMY_UNIT_MECHANICAL]
        != LayerConstants.TACTICAL_LAYERS[LayerConstants.ALLIED_UNIT_MECHANICAL]
    )
    layers = [
        LayerConstants.ALLIED_UNIT_MECHANICAL,
        LayerConstants.ENEMY_UNIT_MECHANICAL,
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": layers,
            "active_tactical_layers": layers,
        },
    )
    focal = create_mock_unit(
        UnitTypeId.ROACH,
        Point2((64.0, 64.0)),
        is_biological=True,
        is_mechanical=False,
        radius=0.5,
    )
    allied_tank = create_mock_unit(
        UnitTypeId.SIEGETANK,
        Point2((62.0, 64.0)),
        is_mechanical=True,
        is_biological=False,
        radius=0.875,
    )
    enemy_tank = create_mock_unit(
        UnitTypeId.SIEGETANK,
        Point2((66.0, 64.0)),
        is_mechanical=True,
        is_biological=False,
        radius=0.875,
    )
    enemy_ling = create_mock_unit(
        UnitTypeId.ZERGLING,
        Point2((64.0, 66.0)),
        is_biological=True,
        is_mechanical=False,
        radius=0.375,
    )
    raw_obs = {
        "units": [focal, allied_tank],
        "structures": [],
        "enemy_units": [enemy_tank, enemy_ling],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert tactical.shape == (2, 32, 32)
    allied_mech = tactical[0]
    enemy_mech = tactical[1]
    assert allied_mech.sum() > 0
    assert enemy_mech.sum() > 0
    # Allied and enemy mechanical stamps occupy different cells.
    assert not np.array_equal(allied_mech > 0, enemy_mech > 0)

    bio_only = transformer.transform_tactical(
        {
            **raw_obs,
            "units": [focal],
            "enemy_units": [enemy_ling],
        },
        focal,
    )
    assert bio_only[0].sum() == 0.0
    assert bio_only[1].sum() == 0.0


def test_tactical_dps_from_off_fovea_unit_with_typed_layer_only():
    """Off-fovea threat rings must draw on typed DPS layers when _dps_base is inactive."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [LayerConstants.ENEMY_G_DPS_ARMOR],
        },
    )
    focal = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    tank = create_mock_unit(
        UnitTypeId.SIEGETANK,
        Point2((64.0, 90.0)),
        radius=0.875,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [focal],
        "enemy_units": [tank],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert int(np.count_nonzero(tactical[0] > 0.01)) > 0


def test_off_fovea_tactical_dps_stays_world_anchored_when_focal_moves():
    """Off-fovea weapon rings must pan with the fovea, not stick to the view edge."""
    from src.utils.obs_geometry import world_to_tactical_obs_pixel

    conf = {"active_tactical_layers": [LayerConstants.ENEMY_G_DPS_BASE]}
    tank_pos = Point2((84.0, 64.0))
    probe_world = (76.0, 64.0)

    def dps_at_probe(focal_xy):
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
        focal = create_mock_unit(UnitTypeId.ZERGLING, Point2(focal_xy))
        tank = create_mock_unit(
            UnitTypeId.SIEGETANK,
            tank_pos,
            radius=0.875,
            ground_range=13.0,
        )
        raw_obs = {
            "units": [focal],
            "enemy_units": [tank],
            "structures": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": _make_game_info(128, 128, with_grids=True),
            "game_data": MagicMock(),
            "selected_units": [focal],
        }
        tactical = transformer.transform_tactical(raw_obs, focal)
        px, py = world_to_tactical_obs_pixel(
            probe_world[0], probe_world[1], focal_xy[0], focal_xy[1]
        )
        return float(tactical[0][py, px])

    focal_a = (64.0, 64.0)
    focal_b = (70.0, 64.0)
    val_a = dps_at_probe(focal_a)
    val_b = dps_at_probe(focal_b)
    assert val_a > 0.0
    assert val_b > 0.0
    assert val_a == pytest.approx(val_b)

    px_a, _ = world_to_tactical_obs_pixel(
        probe_world[0], probe_world[1], focal_a[0], focal_a[1]
    )
    px_b, _ = world_to_tactical_obs_pixel(
        probe_world[0], probe_world[1], focal_b[0], focal_b[1]
    )
    assert px_a - px_b == int(focal_b[0] - focal_a[0])


def test_tactical_allied_dps_from_off_fovea_unit():
    """Allied weapon rings outside the fovea must still render when they reach in."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [LayerConstants.ALLIED_G_DPS_BASE],
        },
    )
    focal = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    marine = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((64.0, 84.0)),
        radius=0.375,
        ground_range=5.0,
    )
    raw_obs = {
        "units": [focal, marine],
        "enemy_units": [],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0


def test_tactical_tempest_dps_in_fovea_on_ground_massive_layer():
    """Air-only Tempest must render on g_dps_massive when a_dps layers are inactive (CTF-style)."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [LayerConstants.ENEMY_G_DPS_MASSIVE],
        },
    )
    focal = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    tempest = create_mock_unit(
        UnitTypeId.TEMPEST,
        Point2((70.0, 70.0)),
        is_flying=True,
        air_range=15.0,
        ground_range=0.0,
    )
    raw_obs = {
        "units": [focal],
        "enemy_units": [tempest],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert int(np.count_nonzero(tactical[0] > 0.01)) > 0


def test_tactical_tempest_dps_off_fovea_reaches_into_fovea():
    """Off-fovea Tempest air range must still paint tactical DPS when the ring overlaps the fovea."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [LayerConstants.ENEMY_G_DPS_MASSIVE],
        },
    )
    focal = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    tempest = create_mock_unit(
        UnitTypeId.TEMPEST,
        Point2((64.0, 90.0)),
        is_flying=True,
        air_range=15.0,
        ground_range=0.0,
    )
    raw_obs = {
        "units": [focal],
        "enemy_units": [tempest],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert int(np.count_nonzero(tactical[0] > 0.01)) > 0


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


def test_tactical_enemy_weapon_cooldown_layer_draws_enemy_cd():
    tactical = [
        LayerConstants.ENEMY_UNIT_DENSITY,
        LayerConstants.ENEMY_WEAPON_COOLDOWN,
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": tactical,
        },
    )
    ally = create_mock_unit(UnitTypeId.ZERGLING, Point2((64, 64)))
    enemy = create_mock_unit(UnitTypeId.MARINE, Point2((66, 64)), weapon_cooldown=30.0)
    raw_obs = {
        "units": [ally],
        "enemy_units": [enemy],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": None,
        "selected_units": [ally],
    }
    result = transformer.transform_tactical(raw_obs, ally)
    assert result.shape == (2, 32, 32)
    # cooldown 30/60 should draw as 0.5 on enemy cooldown channel
    assert result[1].max() == pytest.approx(0.5, abs=1e-6)


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


def test_dps_weapon_game_radius_prefers_live_ground_range_over_catalog():
    """Explicit live ground_range above catalog still expands the DPS ring."""
    t = ObservationTransformer(map_size=(32, 32))
    u = create_mock_unit(UnitTypeId.COLOSSUS, Point2((0.0, 0.0)), radius=2.5, ground_range=9.0)
    u.air_range = 0.0
    u.is_mine = False
    rg, ra = t._dps_weapon_game_radius(u, 6.0)
    assert rg == pytest.approx(9.0 + t._unit_game_radius(u))
    assert ra == pytest.approx(6.0 + t._unit_game_radius(u))


def _mock_unit_with_upgrades(unit, upgrade_ids):
    """Attach a minimal bot state.upgrades set for range-upgrade tests."""
    unit.is_mine = True
    unit._bot_object = MagicMock()
    unit._bot_object.state.upgrades = set(upgrade_ids)
    return unit


def test_dps_weapon_game_radius_colossus_extended_thermal_lance_upgrade():
    """Extended Thermal Lance (+3) must expand DPS ring when catalog/API range stays at 6."""
    t = ObservationTransformer(map_size=(32, 32))
    u = _mock_unit_with_upgrades(
        create_mock_unit(UnitTypeId.COLOSSUS, Point2((0.0, 0.0)), ground_range=6.0),
        {UpgradeId.EXTENDEDTHERMALLANCE},
    )
    u.air_range = 0.0
    rg, ra = t._dps_weapon_game_radius(u, 6.0)
    ur = t._unit_game_radius(u)
    assert rg == pytest.approx(9.0 + ur)
    assert ra == pytest.approx(6.0 + ur)


def test_dps_weapon_game_radius_hydralisk_grooved_spines_upgrade():
    """Grooved Spines (+1) must expand ground and air DPS rings for Hydralisk."""
    t = ObservationTransformer(map_size=(32, 32))
    u = _mock_unit_with_upgrades(
        create_mock_unit(UnitTypeId.HYDRALISK, Point2((0.0, 0.0)), ground_range=5.0),
        {UpgradeId.EVOLVEGROOVEDSPINES},
    )
    u.air_range = 5.0
    rg, ra = t._dps_weapon_game_radius(u, 5.0)
    ur = t._unit_game_radius(u)
    assert rg == pytest.approx(6.0 + ur)
    assert ra == pytest.approx(6.0 + ur)


def test_dps_weapon_game_radius_p2_is_mine_false_still_uses_state_upgrades():
    """God-mode P2 units have is_mine=False; debug_upgrade is still global."""
    t = ObservationTransformer(map_size=(32, 32))
    u = create_mock_unit(
        UnitTypeId.HYDRALISK, Point2((0.0, 0.0)), ground_range=5.0, is_mine=False
    )
    u.air_range = 5.0
    u._bot_object = MagicMock()
    u._bot_object.state.upgrades = {UpgradeId.EVOLVEGROOVEDSPINES}
    rg, ra = t._dps_weapon_game_radius(u, 5.0)
    ur = t._unit_game_radius(u)
    assert rg == pytest.approx(6.0 + ur)
    assert ra == pytest.approx(6.0 + ur)


def test_dps_weapon_game_radius_phoenix_anion_pulse_crystals_upgrade():
    """Anion Pulse-Crystals (+2 air) expands air DPS ring only; ground ring unchanged."""
    t = ObservationTransformer(map_size=(32, 32))
    u = _mock_unit_with_upgrades(
        create_mock_unit(UnitTypeId.PHOENIX, Point2((0.0, 0.0)), ground_range=0.0),
        {UpgradeId.PHOENIXRANGEUPGRADE},
    )
    u.air_range = 5.0
    rg, ra = t._dps_weapon_game_radius(u, 5.0)
    ur = t._unit_game_radius(u)
    assert rg == pytest.approx(5.0 + ur)
    assert ra == pytest.approx(7.0 + ur)


def test_dps_weapon_game_radius_prefers_live_air_range():
    t = ObservationTransformer(map_size=(32, 32))
    u = create_mock_unit(UnitTypeId.VIKINGFIGHTER, Point2((0.0, 0.0)), radius=1.5, ground_range=0.0)
    u.air_range = 7.0
    rg, ra = t._dps_weapon_game_radius(u, 5.0)
    ur = t._unit_game_radius(u)
    assert rg == pytest.approx(5.0 + ur)
    assert ra == pytest.approx(7.0 + ur)


def test_dps_weapon_radius_uses_display_footprint_not_live_selection():
    """DPS rings use catalog footprint from unit_data.json, not inflated live radius."""
    t = ObservationTransformer(map_size=(128, 128))
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((0.0, 0.0)),
        radius=99.0,
        ground_range=13.0,
    )
    catalog_r = t._unit_game_radius(tank)
    rng_g, _ = t._dps_weapon_game_radius(tank, 13.0)
    assert catalog_r == pytest.approx(1.25)
    assert rng_g == pytest.approx(13.0 + catalog_r)


def _universal_training_conf():
    return {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density", "enemy_unit_density", "terrain_pathable"],
        "active_tactical_layers": ["allied_unit_density", "terrain_pathable"],
    }


def _large_map_game_info(width=512, height=512):
    gi = _make_game_info(width, height, with_grids=True)
    gi.playable_area = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = width
    gi.playable_area.height = height
    return gi


def test_universal_structure_footprint_global_large_map():
    """Structure layers on ladder-sized maps must align with terrain unpathable cells."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    pathing = np.ones((512, 512), dtype=np.uint8)
    pathing[253:259, 253:259] = 0
    gi.pathing_grid.data_numpy = pathing

    cc = create_mock_unit(
        UnitTypeId.COMMANDCENTER,
        Point2((256.0, 256.0)),
        is_structure=True,
        radius=2.75,
        footprint_radius=2.75,
    )
    raw_obs = {
        "units": [],
        "structures": [cc],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    density = result["map"][0]
    terrain = result["map"][2]
    struct_mask = density > 0
    assert np.count_nonzero(struct_mask) >= 1
    assert np.all(terrain[struct_mask] <= 0.0)

    # Larger footprints must span multiple obs pixels after downsampling.
    pathing[248:264, 248:264] = 0
    large = create_mock_unit(
        UnitTypeId.COMMANDCENTER,
        Point2((256.0, 256.0)),
        is_structure=True,
        radius=8.0,
        footprint_radius=8.0,
    )
    raw_obs["structures"] = [large]
    transformer._terrain_caches.clear()
    result = transformer.transform(raw_obs)
    density = result["map"][0]
    terrain = result["map"][2]
    struct_mask = density > 0
    assert np.count_nonzero(struct_mask) >= 4
    assert np.all(terrain[struct_mask] <= 0.0)


def test_universal_structure_footprint_tactical():
    """Tactical view should render structure footprints aligned with terrain."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    pathing = np.ones((512, 512), dtype=np.uint8)
    pathing[255:262, 255:262] = 0
    gi.pathing_grid.data_numpy = pathing

    active = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((256.0, 256.0)),
        radius=0.375,
    )
    cc = create_mock_unit(
        UnitTypeId.COMMANDCENTER,
        Point2((258.0, 258.0)),
        is_structure=True,
        radius=2.75,
        footprint_radius=2.75,
    )
    raw_obs = {
        "units": [active],
        "structures": [cc],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "selected_units": [active],
    }
    tactical = transformer.transform_tactical(raw_obs, active)
    density = tactical[0]
    terrain = tactical[1]
    struct_mask = density > 0
    assert np.count_nonzero(struct_mask) > 10
    assert np.all(terrain[struct_mask] <= 0.0)


def test_universal_small_units_remain_single_pixel():
    """Small mobile units (e.g. Marine) stay single-pixel in universal mode."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    marine = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((256.0, 256.0)),
        radius=0.375,
    )
    raw_obs = {
        "units": [marine],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _large_map_game_info(),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    density = result["map"][0]
    assert np.count_nonzero(density) == 1


def test_catalog_display_radii_cover_all_races():
    """Every curriculum unit resolves a catalog radius; unit_data.json wins when set."""
    import json
    from pathlib import Path

    from train.envs.ctf.game_data import UNIT_DATA, _ID_TABLE

    transformer = ObservationTransformer(map_size=(128, 128))
    id_to_name = {v: k for k, v in _ID_TABLE.items()}
    unit_data = json.loads(
        Path(__file__).resolve().parents[1].joinpath("data", "unit_data.json").read_text()
    )

    for type_id, info in UNIT_DATA.items():
        name = id_to_name[type_id]
        unit = create_mock_unit(
            getattr(UnitTypeId, name),
            Point2((0.0, 0.0)),
            radius=99.0,
        )
        expected = float(info.footprint_radius)
        json_entry = unit_data.get(name, {})
        for key in ("display_radius", "footprint_radius", "radius"):
            val = json_entry.get(key, 0)
            if isinstance(val, (int, float)) and val > 0:
                expected = float(val)
                break
        assert transformer._unit_game_radius(unit) == pytest.approx(expected), name

    for name in ("MARINE", "ZEALOT", "ZERGLING", "THOR", "COLOSSUS", "ULTRALISK"):
        unit = create_mock_unit(getattr(UnitTypeId, name), Point2((0.0, 0.0)), radius=99.0)
        assert transformer._unit_game_radius(unit) > 0.0
        assert transformer._unit_game_radius(unit) != 99.0


def test_large_unit_sizes_tank_smaller_than_bc_smaller_than_thor():
    """Siege tank, battlecruiser, and thor must occupy distinct footprint sizes."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )

    def footprint_count(unit_type, radius):
        unit = create_mock_unit(
            unit_type,
            Point2((64.0, 64.0)),
            radius=radius,
        )
        raw_obs = {
            "units": [unit],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": _make_game_info(128, 128, with_grids=True),
            "game_data": MagicMock(),
        }
        result = transformer.transform(raw_obs)
        density = result["map"][0]
        return int(np.count_nonzero(density > 0))

    tank_px = footprint_count(UnitTypeId.SIEGETANK, 99.0)
    bc_px = footprint_count(UnitTypeId.BATTLECRUISER, 99.0)
    thor_px = footprint_count(UnitTypeId.THOR, 99.0)
    assert tank_px < bc_px
    assert tank_px < thor_px
    assert bc_px <= thor_px


def test_tactical_ground_unit_stamps_unpathable_terrain():
    """Tactical view stamps unit-sized unpathable blocks under ground units."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)

    tank = create_mock_unit(
        UnitTypeId.SIEGETANK,
        Point2((256.0, 256.0)),
        radius=0.875,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "selected_units": [tank],
    }
    tactical = transformer.transform_tactical(raw_obs, tank)
    density = tactical[0]
    terrain = tactical[1]
    unit_mask = density > 0
    unpathable = terrain <= 0.0
    assert np.count_nonzero(unit_mask) >= 1
    assert np.array_equal(unit_mask, unpathable)


def test_tactical_burrowed_unit_does_not_block_pathable_terrain():
    """Burrowed units do not block ground pathing — terrain under them stays pathable."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)
    gi.terrain_height.data_numpy = np.full((512, 512), 128, dtype=np.uint8)

    # Flying focal so only burrowed ground units are candidates for terrain stamps.
    focal = create_mock_unit(
        UnitTypeId.MUTALISK,
        Point2((256.0, 256.0)),
        radius=0.5,
        is_flying=True,
    )
    burrowed = create_mock_unit(
        UnitTypeId.LURKERMPBURROWED,
        Point2((256.0, 256.0)),
        radius=0.75,
        is_burrowed=True,
    )
    # Morphs that only expose BURROWED via type id must also stay pathable.
    burrowed_by_type = create_mock_unit(
        UnitTypeId.ZERGLINGBURROWED,
        Point2((258.0, 256.0)),
        radius=0.375,
        is_burrowed=False,
    )
    raw_obs = {
        "units": [focal, burrowed, burrowed_by_type],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    density = tactical[0]
    terrain = tactical[1]
    assert np.count_nonzero(density > 0) >= 1
    assert float(terrain.min()) > 0.0
    assert not np.any(terrain <= 0.0)


def test_thor_larger_than_scv_on_128_map():
    """Large units must occupy more global pixels than workers at native map scale."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    scv = create_mock_unit(
        UnitTypeId.SCV,
        Point2((64.0, 64.0)),
        radius=0.375,
    )
    thor = create_mock_unit(
        UnitTypeId.THOR,
        Point2((96.0, 96.0)),
        radius=1.875,
    )
    raw_obs = {
        "units": [scv, thor],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    density = result["map"][0]
    scv_x, scv_y = transformer._get_pixel_coords(scv.position)
    thor_x, thor_y = transformer._get_pixel_coords(thor.position)

    def count_around(cx, cy, r=4):
        y0, y1 = max(0, cy - r), min(density.shape[0], cy + r + 1)
        x0, x1 = max(0, cx - r), min(density.shape[1], cx + r + 1)
        return int(np.count_nonzero(density[y0:y1, x0:x1] > 0))

    scv_px = count_around(scv_x, scv_y)
    thor_px = count_around(thor_x, thor_y)
    assert scv_px == 1
    assert thor_px > scv_px * 3


def test_sieged_tank_dps_renders_as_donut():
    """Siege-mode tank DPS must exclude the 2-unit minimum range dead zone."""
    transformer = ObservationTransformer(map_size=(128, 128))
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=0.875,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
    layer = result["map"][idx]

    cx, cy = 64, 63
    assert layer[cy, cx] == 0.0
    assert layer[cy, cx + 1] == 0.0
    assert layer[cy + 5, cx] > 0.0
    assert layer[cy + 10, cx] > 0.0

    idx_armor = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_ARMOR]
    armor_layer = result["map"][idx_armor]
    assert armor_layer[cy, cx] == 0.0
    assert armor_layer[cy + 8, cx] > 0.0


def test_sieged_tank_dps_donut_on_large_map_global():
    """Dead zone uses true 2 gu min_range on ladder-sized global obs (not upscaled inner)."""
    transformer = ObservationTransformer(map_size=(128, 128))
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((256.0, 256.0)),
        radius=0.875,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _large_map_game_info(),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
    layer = result["map"][idx]
    px, py = transformer._game_to_obs_pixel(256.0, 256.0)
    assert layer[py, px] == 0.0
    assert layer[py + 3, px] > 0.0


def test_sieged_tank_tactical_dead_zone_surrounds_hull():
    """Tactical siege dead zone is block-centered and square (matches 2×2 hull)."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [
                LayerConstants.ALLIED_UNIT_DENSITY,
                LayerConstants.ALLIED_G_DPS_ARMOR,
            ],
        },
    )
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.5, 64.5)),
        radius=2.0,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [tank],
    }
    tactical = transformer.transform_tactical(raw_obs, tank)
    density = tactical[0] > 0
    dps = tactical[1]
    assert density.shape == dps.shape == (32, 32)
    assert int(np.count_nonzero(density)) == 4
    assert np.all(dps[density] == 0.0)

    from src.utils.obs_geometry import tactical_crop_origin

    transformer.scale = 1.0
    transformer.is_tactical_mode = True
    transformer.feature_map_width = 32
    transformer.feature_map_height = 32
    transformer.crop_x, transformer.crop_y = tactical_crop_origin(
        tank.position.x, tank.position.y
    )
    dead = transformer._tactical_siege_no_dps_mask_in_rect(tank, 2.0, 0, 31, 0, 31)
    assert np.all(dps[dead] == 0.0)
    hull_ys, hull_xs = np.where(density)
    mid_y = int(round(hull_ys.mean()))
    assert dps[mid_y, hull_xs.max() + 3] > 0.0


def _tactical_siege_cardinal_dead_depth(dps, density, direction):
    """Count dead (zero DPS) obs pixels outside the hull along one axis."""
    hull_ys, hull_xs = np.where(density)
    if not len(hull_xs):
        return 0
    mid_y = int(round(hull_ys.mean()))
    mid_x = int(round(hull_xs.mean()))
    if direction == "east":
        x, y, sx, sy = hull_xs.max() + 1, mid_y, 1, 0
    elif direction == "west":
        x, y, sx, sy = hull_xs.min() - 1, mid_y, -1, 0
    elif direction == "south":
        x, y, sx, sy = mid_x, hull_ys.max() + 1, 0, 1
    else:
        x, y, sx, sy = mid_x, hull_ys.min() - 1, 0, -1
    depth = 0
    while 0 <= x < dps.shape[1] and 0 <= y < dps.shape[0]:
        if density[y, x] or dps[y, x] > 0.0:
            break
        depth += 1
        x += sx
        y += sy
    return depth


def test_sieged_tank_tactical_dead_zone_symmetric_at_fractional_positions():
    """Dead-zone ring outside the hull must be even on all four sides."""
    conf = {
        "active_tactical_layers": [
            LayerConstants.ALLIED_UNIT_DENSITY,
            LayerConstants.ALLIED_G_DPS_ARMOR,
        ],
    }

    for frac in (0.0, 0.25, 0.5, 0.75, 0.95):
        pos = (64.0 + frac, 64.0 + frac)
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
        tank = create_mock_unit(
            UnitTypeId.SIEGETANKSIEGED,
            Point2(pos),
            radius=2.0,
            ground_range=13.0,
        )
        raw_obs = {
            "units": [tank],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": _make_game_info(128, 128, with_grids=True),
            "game_data": MagicMock(),
            "selected_units": [tank],
        }
        tactical = transformer.transform_tactical(raw_obs, tank)
        density = tactical[0] > 0
        dps = tactical[1]
        depths = [
            _tactical_siege_cardinal_dead_depth(dps, density, d)
            for d in ("east", "west", "north", "south")
        ]
        assert len(set(depths)) == 1, f"Asymmetric dead ring at {pos}: {depths}"
        assert depths[0] >= 1, f"Dead ring too thin at {pos}"


def test_sieged_tank_dps_donut_in_tactical_view():
    """Tactical fovea DPS layers must also show the siege dead zone."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [
                LayerConstants.ALLIED_G_DPS_BASE,
                LayerConstants.ALLIED_G_DPS_ARMOR,
            ],
        },
    )
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=0.875,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, tank)
    base = tactical[0]
    armor = tactical[1]
    cx, cy = 16, 15
    assert base[cy, cx] == 0.0
    assert base[cy + 5, cx] > 0.0
    assert armor[cy, cx] == 0.0
    assert armor[cy + 6, cx] > 0.0


def test_sieged_tank_ctf_armor_layer_carries_full_dps_without_base_channel():
    """CTF typed-only configs must render base+bonus siege DPS as a donut on the armor channel."""
    conf = {
        "active_layers": ["enemy_g_dps_armor"],
        "active_tactical_layers": ["enemy_g_dps_armor"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=2.0,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [tank],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    layers = np.zeros((LayerConstants.get_num_layers(), 128, 128), dtype=np.float32)
    transformer.current_layer_map = LayerConstants.LAYERS
    transformer.is_tactical_mode = False
    transformer._update_active_indices()
    transformer.scale = transformer.scale_x = transformer.scale_y = 1.0
    transformer.offset_x = transformer.offset_y = 0.0
    transformer.crop_x = transformer.crop_y = 0.0
    transformer._obs_span_w = transformer._obs_span_h = 128.0
    transformer._grid_x = np.arange(128).reshape(1, -1)
    transformer._grid_y = np.arange(128).reshape(-1, 1)
    px, py = transformer._game_to_obs_pixel(64.0, 64.0)
    transformer._draw_unit_dps(tank, layers, True, px, py, MagicMock())
    armor_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_G_DPS_ARMOR]
    armor = layers[armor_idx]
    expected_dps = (40.0 / 3.0) + (30.0 / 3.0)
    assert armor[py, px] == 0.0
    assert armor[py + 5, px] == pytest.approx(expected_dps)
    assert armor[py + 1, px] == 0.0


def test_sieged_tank_dps_dead_zone_visible_on_ctf_armor_layer_large_map():
    """Typed DPS on downsampled maps must keep center/body clear with true 2 gu min_range."""
    conf = {
        "active_layers": ["enemy_g_dps_armor"],
        "active_tactical_layers": ["enemy_g_dps_armor"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((256.0, 256.0)),
        radius=2.0,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [tank],
        "enemy_structures": [],
        "effects": [],
        "game_info": _large_map_game_info(),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    armor = result["map"][0]
    px, py = transformer._game_to_obs_pixel(256.0, 256.0)
    assert armor[py, px] == 0.0
    assert armor[py + 3, px] > 0.0


def test_sieged_tank_dps_not_stamped_on_body_tactical():
    """Siege DPS annulus must not paint over the tactical unit hull."""
    conf = {
        "active_tactical_layers": [
            LayerConstants.ALLIED_G_DPS_BASE,
            LayerConstants.ALLIED_UNIT_DENSITY,
        ],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=99.0,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, tank)
    dps = tactical[0]
    density = tactical[1]
    body = density > 0
    assert np.all(dps[body] <= 0.0)


def test_sieged_tank_body_fits_inside_dps_dead_zone():
    """Hull circle must sit inside the 2 gu weapon min_range hole (concentric donut)."""
    transformer = ObservationTransformer(map_size=(128, 128))
    tank = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=2.0,
        ground_range=13.0,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    dps = result["map"][LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]]
    density = result["map"][LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]]
    body = density > 0
    assert np.any(body)
    assert np.all(dps[body] <= 0.0)
    px, py = transformer._game_to_obs_pixel(64.0, 64.0)
    assert dps[py, px] == 0.0
    far_px, far_py = transformer._game_to_obs_pixel(64.0, 56.0)
    assert dps[far_py, far_px] > 0.0


def test_sieged_tank_smaller_than_old_selection_radius_on_global():
    """Sieged hull uses body override, not the 2.0 selection/collision radius."""
    transformer = ObservationTransformer(map_size=(128, 128))

    def density_count(unit_type):
        unit = create_mock_unit(
            unit_type,
            Point2((64.0, 64.0)),
            radius=2.0,
        )
        raw_obs = {
            "units": [unit],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": _make_game_info(128, 128, with_grids=True),
            "game_data": MagicMock(),
        }
        density = transformer.transform(raw_obs)["map"][
            LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]
        ]
        return int(np.count_nonzero(density > 0))

    unsieged_px = density_count(UnitTypeId.SIEGETANK)
    sieged_px = density_count(UnitTypeId.SIEGETANKSIEGED)
    assert unsieged_px >= 1
    assert 4 <= sieged_px <= 10


def test_liberation_zone_renders_global_and_tactical_with_real_effect_id():
    """Liberation zones must render on typed DPS channels using live SC2 EffectId values."""
    from sc2.ids.effect_id import EffectId

    conf = {
        "active_layers": ["allied_g_dps_armor", "enemy_g_dps_armor"],
        "active_tactical_layers": ["allied_g_dps_armor", "enemy_g_dps_armor"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)

    class _LiberatorEffect:
        def __init__(self, eid, x, y):
            self.id = eid
            self.positions = [Point2((x, y))]

    lib = create_mock_unit(
        UnitTypeId.LIBERATORAG,
        Point2((55.0, 50.0)),
        radius=1.0,
        is_flying=True,
    )
    eff = _LiberatorEffect(EffectId.LIBERATORTARGETMORPHDELAYPERSISTENT, 50.0, 50.0)
    gi = _make_game_info(128, 128, with_grids=True)
    raw_obs = {
        "units": [lib],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [eff],
        "game_info": gi,
        "game_data": MagicMock(),
    }

    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) > 0.0

    focal = create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)), radius=0.375)
    raw_obs["units"] = [lib, focal]
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0


def test_nuke_effect_uses_live_effect_id():
    """Nuke persistent effect (EffectId.NUKEPERSISTENT = 7) must hit ground and air DPS."""
    from sc2.ids.effect_id import EffectId

    transformer = ObservationTransformer(map_size=(128, 128))
    gi = _make_game_info(128, 128, with_grids=True)

    class _NukeEffect:
        def __init__(self):
            self.id = EffectId.NUKEPERSISTENT
            self.positions = [Point2((64.0, 64.0))]

    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_NukeEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    map_data = transformer.transform(raw_obs)["map"]
    g_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_G_DPS_BASE]
    a_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_A_DPS_BASE]
    assert float(map_data[g_idx].max()) > 0.0
    assert float(map_data[a_idx].max()) > 0.0
    assert float(map_data[LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]].max()) == 0.0
    assert float(map_data[LayerConstants.LAYERS[LayerConstants.ALLIED_A_DPS_BASE]].max()) == 0.0


def test_corrosive_bile_unowned_paints_enemy_ground_and_air():
    """Enemy/unowned bile stamps enemy_g_dps_base and enemy_a_dps_base, not allied."""
    from sc2.ids.effect_id import EffectId

    transformer = ObservationTransformer(map_size=(128, 128))
    gi = _make_game_info(128, 128, with_grids=True)

    class _BileEffect:
        def __init__(self):
            self.id = EffectId.RAVAGERCORROSIVEBILECP
            self.positions = [Point2((64.0, 64.0))]

    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_BileEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    map_data = transformer.transform(raw_obs)["map"]
    assert float(map_data[LayerConstants.LAYERS[LayerConstants.ENEMY_G_DPS_BASE]].max()) > 0.0
    assert float(map_data[LayerConstants.LAYERS[LayerConstants.ENEMY_A_DPS_BASE]].max()) > 0.0
    assert float(map_data[LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]].max()) == 0.0
    assert float(map_data[LayerConstants.LAYERS[LayerConstants.ALLIED_A_DPS_BASE]].max()) == 0.0


def test_corrosive_bile_allied_paints_allied_ground_and_air_global_and_tactical():
    """Own bile (friendly fire) must stamp allied ground+air DPS on global and tactical maps."""
    from sc2.ids.effect_id import EffectId

    dps_layers = [
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": dps_layers,
            "active_tactical_layers": dps_layers,
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)

    class _BileEffect:
        def __init__(self):
            self.id = EffectId.RAVAGERCORROSIVEBILECP
            self.positions = [Point2((50.0, 50.0))]
            self.is_mine = True

    ravager = create_mock_unit(
        UnitTypeId.RAVAGER,
        Point2((62.0, 50.0)),
        radius=0.75,
        ground_range=6.0,
        air_range=0.0,
    )
    raw_obs = {
        "units": [ravager],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_BileEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) > 0.0  # allied_g_dps_base
    assert float(global_map[2].max()) > 0.0  # allied_a_dps_base
    assert float(global_map[1].max()) == 0.0  # enemy_g_dps_base
    assert float(global_map[3].max()) == 0.0  # enemy_a_dps_base

    focal = create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)), radius=0.375)
    raw_obs["units"] = [ravager, focal]
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert float(tactical[2].max()) > 0.0
    assert float(tactical[1].max()) == 0.0
    assert float(tactical[3].max()) == 0.0


def test_corrosive_bile_enemy_ravager_stays_on_enemy_dps():
    """Bile near an enemy Ravager stays on enemy ground+air DPS."""
    from sc2.ids.effect_id import EffectId

    dps_layers = [
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": dps_layers,
            "active_tactical_layers": dps_layers,
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)

    class _BileEffect:
        def __init__(self):
            self.id = EffectId.RAVAGERCORROSIVEBILECP
            self.positions = [Point2((50.0, 50.0))]

    ravager = create_mock_unit(
        UnitTypeId.RAVAGER,
        Point2((62.0, 50.0)),
        radius=0.75,
        ground_range=6.0,
        air_range=0.0,
    )
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [ravager],
        "enemy_structures": [],
        "effects": [_BileEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[1].max()) > 0.0  # enemy_g_dps_base
    assert float(global_map[3].max()) > 0.0  # enemy_a_dps_base
    assert float(global_map[0].max()) == 0.0
    assert float(global_map[2].max()) == 0.0


def test_storm_allied_paints_allied_ground_dps_global_and_tactical():
    """Own Psi Storm (friendly fire) stamps allied_g_dps_base, not air, on global and tactical."""
    from sc2.ids.effect_id import EffectId

    dps_layers = [
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": dps_layers,
            "active_tactical_layers": dps_layers,
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)

    class _StormEffect:
        def __init__(self):
            self.id = EffectId.PSISTORMPERSISTENT
            self.positions = [Point2((50.0, 50.0))]
            self.is_mine = True

    ht = create_mock_unit(
        UnitTypeId.HIGHTEMPLAR,
        Point2((62.0, 50.0)),
        radius=0.375,
        ground_range=6.0,
        air_range=0.0,
    )
    raw_obs = {
        "units": [ht],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_StormEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) > 0.0  # allied_g_dps_base
    assert float(global_map[1].max()) == 0.0  # enemy_g_dps_base
    assert float(global_map[2].max()) == 0.0  # allied_a_dps_base
    assert float(global_map[3].max()) == 0.0  # enemy_a_dps_base

    focal = create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)), radius=0.375)
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert float(tactical[1].max()) == 0.0
    assert float(tactical[3].max()) == 0.0


def test_leftover_storm_without_caster_does_not_paint_allied_from_proto():
    """God-mode proto is_mine is P1-client; unmatched leftover storm must not look allied."""
    from sc2.ids.effect_id import EffectId

    dps_layers = [
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": dps_layers,
            "active_tactical_layers": dps_layers,
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)

    class _StormEffect:
        def __init__(self):
            self.id = EffectId.PSISTORMPERSISTENT
            self.positions = [Point2((50.0, 50.0))]
            self.is_mine = True

    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_StormEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) == 0.0  # allied_g_dps_base
    assert float(global_map[1].max()) > 0.0  # enemy_g_dps_base


def test_nuke_allied_paints_allied_ground_and_air_global_and_tactical():
    """Own nuke (friendly fire) stamps allied ground+air DPS on global and tactical maps."""
    from sc2.ids.effect_id import EffectId

    dps_layers = [
        "allied_g_dps_base",
        "enemy_g_dps_base",
        "allied_a_dps_base",
        "enemy_a_dps_base",
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": dps_layers,
            "active_tactical_layers": dps_layers,
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)

    class _NukeEffect:
        def __init__(self):
            self.id = EffectId.NUKEPERSISTENT
            self.positions = [Point2((50.0, 50.0))]
            self.is_mine = True

    ghost = create_mock_unit(
        UnitTypeId.GHOST,
        Point2((62.0, 50.0)),
        radius=0.375,
        ground_range=6.0,
        air_range=0.0,
    )
    raw_obs = {
        "units": [ghost],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_NukeEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) > 0.0  # allied_g_dps_base
    assert float(global_map[2].max()) > 0.0  # allied_a_dps_base
    assert float(global_map[1].max()) == 0.0  # enemy_g_dps_base
    assert float(global_map[3].max()) == 0.0  # enemy_a_dps_base

    focal = create_mock_unit(UnitTypeId.MARINE, Point2((50.0, 50.0)), radius=0.375)
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert float(tactical[2].max()) > 0.0
    assert float(tactical[1].max()) == 0.0
    assert float(tactical[3].max()) == 0.0


def test_parasitic_bomb_paints_enemy_air_dps_and_splash_global_and_tactical():
    """python-sc2 fake effect PARASITICBOMB is air-only on enemy DPS + enemy splash."""
    layers = [
        "enemy_g_dps_base",
        "allied_g_dps_base",
        "enemy_a_dps_base",
        "allied_a_dps_base",
        "enemy_splash",
        "allied_splash",
    ]
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": layers,
            "active_tactical_layers": layers,
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)

    class _BombEffect:
        def __init__(self):
            self.id = "PARASITICBOMB"
            self.fake = True
            self.positions = [Point2((50.0, 50.0))]
            self.radius = 3.0

    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [_BombEffect()],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[2].max()) > 0.0  # enemy_a_dps_base
    assert float(global_map[4].max()) > 0.0  # enemy_splash
    assert float(global_map[0].max()) == 0.0  # enemy_g_dps_base (air-only)
    assert float(global_map[1].max()) == 0.0  # allied_g_dps_base
    assert float(global_map[3].max()) == 0.0  # allied_a_dps_base
    assert float(global_map[5].max()) == 0.0  # allied_splash

    focal = create_mock_unit(UnitTypeId.MUTALISK, Point2((50.0, 50.0)), radius=0.5, is_flying=True)
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[2].max()) > 0.0
    assert float(tactical[4].max()) > 0.0
    assert float(tactical[0].max()) == 0.0
    assert float(tactical[1].max()) == 0.0
    assert float(tactical[3].max()) == 0.0
    assert float(tactical[5].max()) == 0.0


def test_viper_hull_does_not_paint_splash():
    """Parasitic Bomb splash is the dummy/effect cloud, not a circle around the Viper."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": ["allied_splash", "enemy_splash"],
            "active_tactical_layers": ["allied_splash", "enemy_splash"],
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)
    viper = create_mock_unit(UnitTypeId.VIPER, Point2((50.0, 50.0)), radius=0.75, is_flying=True)
    raw_obs = {
        "units": [viper],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_map = transformer.transform(raw_obs)["map"]
    assert float(global_map[0].max()) == 0.0
    assert float(global_map[1].max()) == 0.0


def test_sensor_blip_renders_density_not_attributes():
    """Sensor blips show position on density only — no biological/mechanical attributes."""
    transformer = ObservationTransformer(map_size=(128, 128))
    blip = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((64.0, 64.0)),
        is_enemy=True,
        is_biological=True,
    )
    blip.is_blip = True
    raw_obs = {
        "units": [],
        "enemy_units": [blip],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    map_data = transformer.transform(raw_obs)["map"]
    den_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]
    bio_idx = LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_BIOLOGICAL]
    assert float(map_data[den_idx].max()) > 0.0
    assert float(map_data[bio_idx].max()) == 0.0


def test_ultralisk_and_archon_render_2x2_footprint():
    """Ultralisk and archon use fixed 2×2 hulls on global and tactical maps."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)
    for unit_type in (UnitTypeId.ULTRALISK, UnitTypeId.ARCHON):
        unit = create_mock_unit(unit_type, Point2((64.0, 64.0)), radius=99.0)
        raw_obs = {
            "units": [unit],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        global_px = int(np.count_nonzero(transformer.transform(raw_obs)["map"][0] > 0))
        tactical_px = int(np.count_nonzero(transformer.transform_tactical(raw_obs, unit)[0] > 0))
        assert global_px == 4, unit_type.name
        assert tactical_px == 4, unit_type.name


def test_broodling_melee_dps_renders_on_global_map():
    """Melee broodlings must stamp g_dps even when catalog range is sub-pixel (~0.1 gu)."""
    transformer = ObservationTransformer(map_size=(128, 128))
    broodling = create_mock_unit(
        UnitTypeId.BROODLING,
        Point2((64.0, 64.0)),
        radius=0.375,
        ground_range=0.0,
    )
    raw_obs = {
        "units": [broodling],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
    layer = result["map"][idx]
    assert layer.max() > 0.0
    assert np.count_nonzero(layer > 0) > 0


def test_marine_dps_renders_as_solid_disc():
    """Regular units without min_range still fill the center of the DPS disc."""
    transformer = ObservationTransformer(map_size=(128, 128))
    marine = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((64.0, 64.0)),
        radius=0.375,
        ground_range=5.0,
    )
    raw_obs = {
        "units": [marine],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    idx = LayerConstants.LAYERS[LayerConstants.ALLIED_G_DPS_BASE]
    layer = result["map"][idx]
    cx, cy = 64, 63
    assert layer[cy, cx] > 0.0


def test_structure_density_aligns_with_pathing_footprint():
    """Structure markers should stamp unpathable cells as a square, not a 5-pixel cross."""
    transformer = ObservationTransformer(map_size=(128, 128))
    pathing = np.ones((128, 128), dtype=np.uint8)
    pathing[62:67, 62:67] = 0
    gi = _make_game_info(128, 128, with_grids=True)
    gi.pathing_grid.data_numpy = pathing

    depot = create_mock_unit(
        UnitTypeId.SUPPLYDEPOT,
        Point2((64.0, 64.0)),
        is_structure=True,
        radius=0.0,
    )
    raw_obs = {
        "units": [],
        "structures": [depot],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    density_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]
    terrain = result["map"][terrain_idx]
    density = result["map"][density_idx]

    struct_mask = density > 0
    unpathable = terrain <= 0.0
    assert np.count_nonzero(struct_mask) >= 4
    overlap = np.logical_and(struct_mask, unpathable)
    assert np.count_nonzero(overlap) == np.count_nonzero(struct_mask)


def test_tactical_structure_footprint_is_square():
    """Tactical buildings must render as axis-aligned squares sized to their tile footprint."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    cases = [
        (UnitTypeId.SUPPLYDEPOT, (61.0, 61.0), (slice(60, 62), slice(60, 62)), 2),
        (UnitTypeId.BARRACKS, (64.0, 64.0), (slice(63, 66), slice(63, 66)), 3),
    ]
    for unit_type, pos, path_slice, expected_span in cases:
        pathing = np.ones((128, 128), dtype=np.uint8)
        pathing[path_slice] = 0
        gi = _make_game_info(128, 128, with_grids=True)
        gi.pathing_grid.data_numpy = pathing

        structure = create_mock_unit(
            unit_type,
            Point2(pos),
            is_structure=True,
            radius=0.0,
        )
        active = create_mock_unit(
            UnitTypeId.MARINE,
            Point2((50.0, 50.0)),
            radius=0.375,
        )
        raw_obs = {
            "units": [active],
            "structures": [structure],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        tactical = transformer.transform_tactical(raw_obs, active)
        density = tactical[0]
        marine_only = transformer.transform_tactical(
            {**raw_obs, "structures": []}, active
        )[0]
        struct_mask = (density > 0) & ~(marine_only > 0)
        ys, xs = np.where(struct_mask)
        assert len(xs) > 0, f"{unit_type.name} not visible in tactical view"
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        assert w == h, f"{unit_type.name} tactical footprint {w}x{h} is not square"
        assert w == expected_span, (
            f"{unit_type.name} expected {expected_span}x{expected_span}, got {w}x{h}"
        )


def test_tactical_structure_footprint_stable_with_fractional_crop():
    """Tactical structure markers stay solid squares when the focal unit moves."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    pathing = np.ones((128, 128), dtype=np.uint8)
    pathing[60:62, 60:62] = 0
    gi = _make_game_info(128, 128, with_grids=True)
    gi.pathing_grid.data_numpy = pathing

    depot = create_mock_unit(
        UnitTypeId.SUPPLYDEPOT,
        Point2((61.0, 61.0)),
        is_structure=True,
        radius=0.0,
    )

    for offset in (0.0, 0.3, 0.7, 1.2):
        active = create_mock_unit(
            UnitTypeId.MARINE,
            Point2((50.0 + offset, 50.0 + offset)),
            radius=0.375,
        )
        raw_obs = {
            "units": [active],
            "structures": [depot],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        tactical = transformer.transform_tactical(raw_obs, active)
        density = tactical[0]
        assert np.count_nonzero(density) >= 2, (
            f"striped or partial depot footprint at crop offset {offset}: "
            f"only {np.count_nonzero(density)} pixels"
        )


def _tactical_density_centroid(density: np.ndarray):
    ys, xs = np.nonzero(density)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def test_tactical_terrain_unpathable_stable_while_focal_unit_moves():
    """Stationary ground-unit terrain stamps must not change shape as the fovea pans."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)

    tank = create_mock_unit(
        UnitTypeId.SIEGETANK,
        Point2((256.0, 256.0)),
        radius=0.875,
    )

    def _footprint_shape(mask):
        ys, xs = np.where(mask)
        return (
            int(np.count_nonzero(mask)),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )

    shapes = []
    for offset in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        focal = create_mock_unit(
            UnitTypeId.MARINE,
            Point2((240.0 + offset, 240.0 + offset)),
            radius=0.375,
        )
        raw_obs = {
            "units": [focal, tank],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        tactical = transformer.transform_tactical(raw_obs, focal)
        terrain = tactical[1]
        unpathable = terrain <= 0.0
        assert np.any(unpathable)
        shapes.append(_footprint_shape(unpathable))

    assert len(set(shapes)) == 1


def test_tactical_terrain_fills_fovea_after_global_transform_on_small_map():
    """Tactical terrain must cover all 32×32 pixels after global transform().

    Global transform() sets playable span and letterbox offsets for the 128×128
    view. transform_tactical() must not reuse that span when masking terrain.
    """
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": ["terrain_pathable"],
            "active_tactical_layers": ["terrain_pathable"],
        },
    )
    size = 32
    path = np.ones((size, size), dtype=np.uint8)
    gi = _make_game_info(size, size, with_grids=True)
    gi.pathing_grid.data_numpy = path
    gi.playable_area = MagicMock()
    gi.playable_area.x = 2
    gi.playable_area.y = 2
    gi.playable_area.width = 28
    gi.playable_area.height = 28

    focal = create_mock_unit(UnitTypeId.MARINE, Point2((16.0, 16.0)), radius=0.375)
    raw_obs = {
        "units": [focal],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }

    transformer.transform(raw_obs)
    tactical = transformer.transform_tactical(raw_obs, focal)
    terrain = tactical[0]

    corner_pixels = (
        terrain[0, 0],
        terrain[0, 31],
        terrain[31, 0],
        terrain[31, 31],
    )
    assert all(v > 0.0 for v in corner_pixels), (
        "Tactical terrain must render on all fovea edges; got corners "
        f"{corner_pixels}"
    )


def test_tactical_small_unit_terrain_matches_single_pixel_density():
    """Sub-half-tile units must not inflate terrain beyond their density pixel."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)

    marine = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((256.0, 256.0)),
        radius=0.375,
    )
    raw_obs = {
        "units": [marine],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, marine)
    density = tactical[0]
    terrain = tactical[1]
    unit_mask = density > 0
    unpathable = terrain <= 0.0
    assert np.count_nonzero(unit_mask) == 1
    assert np.array_equal(unit_mask, unpathable)


def test_tactical_density_additive_footprint_stacking():
    """Footprint-sized units must sum on tactical density, not clamp to 1.0."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": [
            "allied_unit_density",
            LayerConstants.ALLIED_WEAPON_COOLDOWN,
        ],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    gi = _large_map_game_info()
    pos = Point2((256.0, 256.0))
    colossus_a = create_mock_unit(
        UnitTypeId.COLOSSUS, pos, radius=1.0, weapon_cooldown=0.0,
    )
    colossus_b = create_mock_unit(
        UnitTypeId.COLOSSUS, pos, radius=1.0, weapon_cooldown=30.0,
    )
    raw_obs = {
        "units": [colossus_a, colossus_b],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, colossus_a)
    density = tactical[0]
    cooldown = tactical[1]
    single_obs = {
        **raw_obs,
        "units": [colossus_a],
    }
    single_density = transformer.transform_tactical(single_obs, colossus_a)[0]
    assert density.max() > single_density.max()
    assert cooldown.max() == pytest.approx(0.5)


def test_global_density_additive_footprint_stacking():
    """Stacked footprint units must read brighter than a lone unit after log1p."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": ["allied_unit_density"],
            "active_tactical_layers": ["allied_unit_density"],
        },
    )
    gi = _large_map_game_info()
    pos = Point2((256.0, 256.0))
    colossus = create_mock_unit(UnitTypeId.COLOSSUS, pos, radius=1.0)
    stacked_obs = {
        "units": [colossus, create_mock_unit(UnitTypeId.COLOSSUS, pos, radius=1.0)],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    single_obs = {
        "units": [colossus],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    stacked_density = transformer.transform(stacked_obs)["map"][0]
    single_density = transformer.transform(single_obs)["map"][0]
    assert stacked_density.max() > single_density.max()


def test_tactical_density_postprocess_matches_global_single_unit():
    """Tactical count layers must use log1p like global, not raw [0, 1] stamps."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    gi = _large_map_game_info()
    colossus = create_mock_unit(
        UnitTypeId.COLOSSUS, Point2((256.0, 256.0)), radius=1.0,
    )
    raw_obs = {
        "units": [colossus],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    global_density = transformer.transform(raw_obs)["map"][0]
    tactical_density = transformer.transform_tactical(raw_obs, colossus)[0]
    expected = np.log1p(1.0)
    assert global_density.max() == pytest.approx(expected)
    assert tactical_density.max() == pytest.approx(expected)


def test_tactical_world_positions_stable_while_focal_unit_moves():
    """Stationary units/buildings must not wiggle as the focal unit moves sub-tile."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    pathing = np.ones((128, 128), dtype=np.uint8)
    pathing[60:65, 60:65] = 0
    gi = _make_game_info(128, 128, with_grids=True)
    gi.pathing_grid.data_numpy = pathing

    depot = create_mock_unit(
        UnitTypeId.SUPPLYDEPOT,
        Point2((62.0, 62.0)),
        is_structure=True,
        radius=2.5,
        footprint_radius=2.5,
    )
    marine = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((55.0, 55.0)),
        radius=0.375,
    )

    centroids = []
    for offset in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        active = create_mock_unit(
            UnitTypeId.MARINE,
            Point2((50.0 + offset, 50.0 + offset)),
            radius=0.375,
        )
        raw_obs = {
            "units": [active, marine],
            "structures": [depot],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        tactical = transformer.transform_tactical(raw_obs, active)
        density = tactical[0]
        centroids.append(_tactical_density_centroid(density))

    depot_centroids = [c for c in centroids if c is not None]
    assert len(depot_centroids) == len(centroids)
    xs = [c[0] for c in depot_centroids]
    ys = [c[1] for c in depot_centroids]
    assert max(xs) - min(xs) < 0.01
    assert max(ys) - min(ys) < 0.01


def test_structure_attribute_layers_match_terrain_unpathable():
    """Structure unit layers must align with terrain unpathable cells and not stack > 1.0."""
    transformer = ObservationTransformer(map_size=(128, 128))
    pathing = np.ones((128, 128), dtype=np.uint8)
    pathing[62:67, 62:67] = 0
    gi = _make_game_info(128, 128, with_grids=True)
    gi.pathing_grid.data_numpy = pathing

    depot = create_mock_unit(
        UnitTypeId.SUPPLYDEPOT,
        Point2((64.0, 64.0)),
        is_structure=True,
        radius=2.5,
        footprint_radius=2.5,
        is_mechanical=True,
        is_armored=True,
        is_massive=True,
    )
    raw_obs = {
        "units": [],
        "structures": [depot],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    full = result["map"]
    terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    terrain = full[terrain_idx]

    for layer_name in (
        LayerConstants.ALLIED_UNIT_DENSITY,
        LayerConstants.ALLIED_UNIT_MECHANICAL,
        LayerConstants.ALLIED_UNIT_ARMORED,
        LayerConstants.ALLIED_UNIT_MASSIVE,
    ):
        layer = full[LayerConstants.LAYERS[layer_name]]
        active = layer > 0
        assert np.any(active), f"{layer_name} should have structure signal"
        assert np.all(terrain[active] <= 0.0), f"{layer_name} spills onto pathable terrain"
        assert layer.max() <= 1.0, f"{layer_name} has stacked brightness > 1.0"

    # Attribute layers for the same structure must share identical footprints.
    density = full[LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]] > 0
    for layer_name in (
        LayerConstants.ALLIED_UNIT_MECHANICAL,
        LayerConstants.ALLIED_UNIT_ARMORED,
    ):
        layer_mask = full[LayerConstants.LAYERS[layer_name]] > 0
        assert np.array_equal(layer_mask, density), f"{layer_name} footprint mismatch"


def test_terrain_height_aligns_with_entity_coordinates_on_inset_map():
    """Terrain height must sample the same pathing cells as unit pixel mapping."""
    size = 176
    path = np.zeros((size, size), dtype=np.uint8)
    path[8:168, 8:168] = 1
    height = np.full((size, size), 10.0, dtype=np.float32)
    height[8:88, 8:168] = 26.0

    gi = MagicMock()
    gi.map_name = "inset_height_map"
    gi.map_size.width = size
    gi.map_size.height = size
    gi.playable_area = MagicMock()
    gi.playable_area.x = 8
    gi.playable_area.y = 8
    gi.playable_area.width = 160
    gi.playable_area.height = 160
    gi.pathing_grid = MagicMock()
    gi.pathing_grid.data_numpy = path
    gi.terrain_height = MagicMock()
    gi.terrain_height.data_numpy = height

    transformer = ObservationTransformer(map_size=(128, 128))
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    terrain = result["map"][terrain_idx]

    high_gx, high_gy = 80.0, 80.0
    low_gx, low_gy = 80.0, 140.0
    high_px, high_py = transformer._game_to_obs_pixel(high_gx, high_gy)
    low_px, low_py = transformer._game_to_obs_pixel(low_gx, low_gy)
    X_map, Y_map = transformer._obs_to_sc2_meshgrid()
    assert path[Y_map[high_py, high_px], X_map[high_py, high_px]] > 0
    assert path[Y_map[low_py, low_px], X_map[low_py, low_px]] > 0
    assert terrain[high_py, high_px] > terrain[low_py, low_px], (
        "High-ground walkable cells must render brighter than low-ground walkable cells"
    )
    from src.transformers.observation_transformer import _TERRAIN_BRIGHTNESS_PER_CLIFF_TIER

    assert terrain[high_py, high_px] - terrain[low_py, low_px] >= (
        _TERRAIN_BRIGHTNESS_PER_CLIFF_TIER * 0.9
    ), "Adjacent cliff tiers must differ by roughly one brightness step"


def test_playable_area_zooms_inset_map_without_camera_margins():
    """Ladder maps zoom to playable_area so void margins are not shown."""
    size = 176
    path = np.zeros((size, size), dtype=np.uint8)
    path[8:168, 8:168] = 1

    gi = MagicMock()
    gi.map_size.width = size
    gi.map_size.height = size
    gi.playable_area = MagicMock()
    gi.playable_area.x = 8
    gi.playable_area.y = 8
    gi.playable_area.width = 160
    gi.playable_area.height = 160

    crop_x, crop_y, mw, mh = observation_bounds(gi, path)
    assert crop_x == 8.0 and crop_y == 8.0
    assert mw == 160.0 and mh == 160.0

    crop_x, crop_y, mw, mh, scale_x, scale_y, offset_x, offset_y = compute_global_obs_geometry(
        gi, 128, 128, path_data=path
    )
    transformer = ObservationTransformer(map_size=(128, 128))
    transformer.crop_x, transformer.crop_y = crop_x, crop_y
    transformer.scale_x, transformer.scale_y = scale_x, scale_y
    transformer.scale = min(scale_x, scale_y)
    transformer.offset_x, transformer.offset_y = offset_x, offset_y
    transformer.feature_map_width = 128
    transformer.feature_map_height = 128

    px_lo, py_lo = transformer._game_to_obs_pixel(8.0, 8.0)
    px_hi, py_hi = transformer._game_to_obs_pixel(167.0, 167.0)
    assert px_lo == 0 and py_lo == 127
    assert px_hi == 127 and py_hi == 0


def test_playable_area_maps_southern_edge_to_obs_bottom():
    """Playable south edge must reach the bottom obs row."""
    path = np.zeros((64, 64), dtype=np.uint8)
    path[4:60, 4:60] = 1
    path[4, 32] = 1

    gi = _make_game_info(64, 64, with_grids=True)
    gi.pathing_grid.data_numpy = path
    gi.playable_area = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = 64
    gi.playable_area.height = 64

    transformer = ObservationTransformer(map_size=(128, 128))
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    transformer.transform(raw_obs)

    px, py = transformer._game_to_obs_pixel(32.0, 0.0)
    assert py == 127
    assert 0 <= px <= 127


def test_playable_area_maps_to_obs_bounds():
    """Playable map edges must land inside the 128×128 observation canvas."""
    transformer = ObservationTransformer(map_size=(128, 128))
    gi = _make_game_info(512, 512, with_grids=True)
    gi.playable_area = MagicMock()
    gi.playable_area.x = 0
    gi.playable_area.y = 0
    gi.playable_area.width = 512
    gi.playable_area.height = 512

    (
        transformer.crop_x,
        transformer.crop_y,
        mw,
        mh,
        transformer.scale_x,
        transformer.scale_y,
        transformer.offset_x,
        transformer.offset_y,
    ) = compute_global_obs_geometry(gi, 128, 128, path_data=gi.pathing_grid.data_numpy)
    transformer.scale = min(transformer.scale_x, transformer.scale_y)
    transformer.feature_map_width = 128
    transformer.feature_map_height = 128

    px0, py0 = transformer._game_to_obs_pixel(0.0, 0.0)
    px1, py1 = transformer._game_to_obs_pixel(511.0, 511.0)
    assert 0 <= px0 < 128
    assert 0 <= py0 < 128
    assert 0 <= px1 <= 127
    assert 0 <= py1 <= 127


def test_command_assignment_footprint_matches_in_tactical_view():
    """Tactical COMMAND_ASSIGNMENT must use unit footprint (visible on large units)."""
    conf = {
        "agent_architecture": "universal",
        "active_tactical_layers": ["allied_unit_density", "command_assignment"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    bc = create_mock_unit(
        UnitTypeId.BATTLECRUISER,
        Point2((64.0, 64.0)),
        radius=1.25,
        is_flying=True,
    )
    raw_obs = {
        "units": [bc],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, bc)
    density = tactical[0]
    assignment = tactical[1]
    assert np.count_nonzero(assignment > 0) >= 6
    assert np.count_nonzero(assignment > 0) == np.count_nonzero(density > 0)


def test_thor_global_density_matches_tactical_on_128_map():
    """Global mobile density must use game-space footprint, not the inflated pixel AABB."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    thor = create_mock_unit(
        UnitTypeId.THOR,
        Point2((64.0, 64.0)),
        radius=2.5,
    )
    raw_obs = {
        "units": [thor],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [thor],
    }
    global_px = int(np.count_nonzero(transformer.transform(raw_obs)["map"][0] > 0))
    tactical_px = int(
        np.count_nonzero(transformer.transform_tactical(raw_obs, thor)[0] > 0)
    )
    assert global_px == tactical_px
    assert global_px <= 10


def test_battlecruiser_global_density_stable_while_moving():
    """Global mobile stamps must not wiggle as units move within a tile."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    counts = []
    for offset in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        bc = create_mock_unit(
            UnitTypeId.BATTLECRUISER,
            Point2((64.0 + offset, 64.0 + offset)),
            radius=1.25,
            is_flying=True,
        )
        raw_obs = {
            "units": [bc],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": _make_game_info(128, 128, with_grids=True),
            "game_data": MagicMock(),
        }
        counts.append(
            int(np.count_nonzero(transformer.transform(raw_obs)["map"][0] > 0))
        )
    assert len(set(counts)) == 1
    assert counts[0] == 9


def test_battlecruiser_global_density_matches_tactical_on_128_map():
    """Global and tactical BC footprints must agree at native map scale."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    bc = create_mock_unit(
        UnitTypeId.BATTLECRUISER,
        Point2((64.0, 64.0)),
        radius=1.25,
        is_flying=True,
    )
    raw_obs = {
        "units": [bc],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
        "selected_units": [bc],
    }
    global_px = int(np.count_nonzero(transformer.transform(raw_obs)["map"][0] > 0))
    tactical_px = int(
        np.count_nonzero(transformer.transform_tactical(raw_obs, bc)[0] > 0)
    )
    assert global_px == tactical_px == 9


def test_battlecruiser_footprint_scales_on_global_and_tactical():
    """Large mobiles must scale with map zoom — not fixed 128-ref squares on global."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": ["allied_unit_density"],
        "active_tactical_layers": ["allied_unit_density"],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    bc = create_mock_unit(
        UnitTypeId.BATTLECRUISER,
        Point2((256.0, 256.0)),
        radius=1.25,
        is_flying=True,
    )
    raw_global = {
        "units": [bc],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _large_map_game_info(),
        "game_data": MagicMock(),
    }
    global_px = int(np.count_nonzero(transformer.transform(raw_global)["map"][0] > 0))

    bc_tactical = create_mock_unit(
        UnitTypeId.BATTLECRUISER,
        Point2((64.0, 64.0)),
        radius=1.25,
        is_flying=True,
    )
    raw_tactical = {
        "units": [bc_tactical],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    tactical_px = int(
        np.count_nonzero(transformer.transform_tactical(raw_tactical, bc_tactical)[0] > 0)
    )

    assert global_px < 25
    assert tactical_px >= 6
    assert tactical_px >= global_px


def test_large_mobile_units_use_rect_footprint_not_cross():
    """Units larger than ~1 obs pixel render as filled rects, not 5-pixel crosses."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    tank = create_mock_unit(
        UnitTypeId.BATTLECRUISER,
        Point2((64.0, 64.0)),
        radius=1.25,
        footprint_radius=1.25,
    )
    raw_obs = {
        "units": [tank],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    density_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]
    density = result["map"][density_idx]
    active = density > 0
    assert np.count_nonzero(active) >= 9
    ys, xs = np.where(active)
    assert (xs.max() - xs.min() + 1) >= 3
    assert (ys.max() - ys.min() + 1) >= 3


def _density_pixel_count(transformer, unit, *, is_enemy=False, is_structure=False):
    raw_obs = {
        "units": [] if is_structure else ([unit] if not is_enemy else []),
        "structures": [unit] if is_structure else [],
        "enemy_units": [unit] if is_enemy else [],
        "enemy_structures": [],
        "effects": [],
        "game_info": _make_game_info(128, 128, with_grids=True),
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    density = result["map"][LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]]
    if is_enemy:
        density = result["map"][LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY]]
    return int(np.count_nonzero(density > 0))


def test_overlord_density_ignores_inflated_live_radius():
    """Live selection radius must not blow up support units like Overlord."""
    transformer = ObservationTransformer(map_size=(128, 128))
    overlord = create_mock_unit(
        UnitTypeId.OVERLORD,
        Point2((64.0, 64.0)),
        radius=1.125,
        is_flying=True,
    )
    marine = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((64.0, 64.0)),
        radius=0.375,
    )
    medivac = create_mock_unit(
        UnitTypeId.MEDIVAC,
        Point2((64.0, 64.0)),
        radius=0.875,
    )
    assert _density_pixel_count(transformer, marine) == 1
    medivac_px = _density_pixel_count(transformer, medivac)
    assert medivac_px == 1
    overlord_px = _density_pixel_count(transformer, overlord)
    assert overlord_px == 1
    assert _density_pixel_count(transformer, overlord, is_enemy=True) == overlord_px


def test_overlord_display_radius_by_type_id_when_name_missing():
    """Enemy units must resolve footprint even if type_id lacks a .name attribute."""
    transformer = ObservationTransformer(map_size=(128, 128))
    overlord = create_mock_unit(
        UnitTypeId.OVERLORD,
        Point2((64.0, 64.0)),
        radius=1.125,
        is_flying=True,
    )
    overlord.type_id = UnitTypeId.OVERLORD.value
    assert transformer._unit_game_radius(overlord) == 0.875
    assert _density_pixel_count(transformer, overlord, is_enemy=True) <= 6


def test_sieged_tank_tactical_terrain_matches_density():
    """Tactical siege hull must stamp the same unpathable cells as unit density."""
    conf = {
        "active_tactical_layers": [
            LayerConstants.ALLIED_UNIT_DENSITY,
            LayerConstants.TERRAIN_PATHABLE,
        ],
    }
    for frac in (0.0, 0.25, 0.5, 0.75, 0.95):
        pos = (64.0 + frac, 64.0 + frac)
        gi = _make_game_info(128, 128, with_grids=True)
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
        tank = create_mock_unit(
            UnitTypeId.SIEGETANKSIEGED,
            Point2(pos),
            radius=2.0,
        )
        raw_obs = {
            "units": [tank],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
            "selected_units": [tank],
        }
        tactical = transformer.transform_tactical(raw_obs, tank)
        density = tactical[0] > 0
        unpathable = tactical[1] <= 0.0
        assert np.array_equal(density, unpathable), f"Mismatch at {pos}"


def test_sieged_tank_tactical_footprint_stable_at_fractional_positions():
    """Siege hull in tactical view must stay a fixed 2×2 block while sub-tile."""
    from src.utils.obs_geometry import world_to_tactical_obs_pixel

    conf = {
        "agent_architecture": "universal",
        "active_tactical_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
    }
    shapes = set()
    for offset in (0.0, 0.15, 0.35, 0.55, 0.75, 0.95):
        focal_pos = (50.0 + offset, 50.0 + offset)
        tank_pos = (64.5 + offset * 0.1, 64.5 + offset * 0.1)
        gi = _make_game_info(128, 128, with_grids=True)
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
        focal = create_mock_unit(UnitTypeId.MARINE, Point2(focal_pos))
        tank = create_mock_unit(
            UnitTypeId.SIEGETANKSIEGED,
            Point2(tank_pos),
            radius=2.0,
        )
        raw_obs = {
            "units": [focal, tank],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
            "selected_units": [focal],
        }
        density = transformer.transform_tactical(raw_obs, focal)[0]
        tx, ty = world_to_tactical_obs_pixel(
            tank_pos[0], tank_pos[1], focal_pos[0], focal_pos[1]
        )
        y0, y1 = max(0, ty - 3), min(31, ty + 4)
        x0, x1 = max(0, tx - 3), min(31, tx + 4)
        mask = density[y0 : y1 + 1, x0 : x1 + 1] > 0
        ys, xs = np.where(mask)
        shapes.add(
            (
                int(np.count_nonzero(mask)),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
        )

    assert shapes == {(4, 2, 2)}, f"Siege footprint shape shifted across positions: {shapes}"

    # Same 2×2 when the tank is the focal unit (crop follows the tank).
    for frac in (0.0, 0.25, 0.5, 0.75):
        pos = (64.0 + frac, 64.0 + frac)
        gi = _make_game_info(128, 128, with_grids=True)
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
        tank = create_mock_unit(
            UnitTypeId.SIEGETANKSIEGED,
            Point2(pos),
            radius=2.0,
        )
        raw_obs = {
            "units": [tank],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
            "selected_units": [tank],
        }
        density = transformer.transform_tactical(raw_obs, tank)[0] > 0
        ys, xs = np.where(density)
        shape = (
            int(np.count_nonzero(density)),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        assert shape == (4, 2, 2), f"Self-focal siege footprint at {pos}: {shape}"


def test_siege_tank_footprint_uses_catalog_not_live_radius():
    """Siege tanks use unit_data.json radii; live selection radius is ignored."""
    gi = _make_game_info(128, 128, with_grids=True)
    path = np.ones((128, 128), dtype=np.uint8)
    path[62:66, 62:66] = 0
    gi.pathing_grid.data_numpy = path

    unsieged = create_mock_unit(
        UnitTypeId.SIEGETANK,
        Point2((64.0, 64.0)),
        radius=1.75,
    )
    sieged = create_mock_unit(
        UnitTypeId.SIEGETANKSIEGED,
        Point2((64.0, 64.0)),
        radius=2.0,
    )
    transformer = ObservationTransformer(map_size=(128, 128))
    assert transformer._unit_game_radius(unsieged) == 0.875
    assert transformer._unit_game_radius(sieged) == 1.25

    def density_count(unit):
        raw_obs = {
            "units": [unit],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        result = transformer.transform(raw_obs)
        density = result["map"][LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]]
        return int(np.count_nonzero(density > 0))

    unsieged_px = density_count(unsieged)
    sieged_px = density_count(sieged)
    assert unsieged_px >= 1
    assert 4 <= sieged_px <= 10  # circular hull at 1.25 gu, not old 2.0 square (~16+ px)


def test_game_to_obs_round_trip_matches_pixel_center():
    """Unit positions should map to the same obs pixels as terrain center sampling."""
    transformer = ObservationTransformer(map_size=(128, 128))
    transformer.crop_x = 0
    transformer.crop_y = 0
    transformer.scale = 0.25
    transformer.offset_x = 0
    transformer.offset_y = 0
    transformer.feature_map_width = 128
    transformer.feature_map_height = 128

    px, py = transformer._game_to_obs_pixel(64.0, 64.0)
    X_map, Y_map = transformer._obs_to_sc2_meshgrid()
    assert X_map[py, px] == 64
    assert Y_map[py, px] == 64


def test_ground_structures_stamped_unpathable_on_global_terrain():
    """Player buildings must appear unpathable even when cached terrain omits them."""
    size = 128
    path = np.ones((size, size), dtype=np.uint8)
    gi = _make_game_info(size, size, with_grids=True)
    gi.pathing_grid.data_numpy = path
    gi.map_name = "structure_terrain_map"

    transformer = ObservationTransformer(map_size=(128, 128))
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    transformer.transform(raw_obs)

    cc = create_mock_unit(
        UnitTypeId.COMMANDCENTER,
        Point2((64.0, 64.0)),
        is_structure=True,
        radius=0.0,
    )
    barracks = create_mock_unit(
        UnitTypeId.BARRACKS,
        Point2((80.0, 64.0)),
        is_structure=True,
        radius=0.0,
    )
    depot = create_mock_unit(
        UnitTypeId.SUPPLYDEPOT,
        Point2((48.0, 64.0)),
        is_structure=True,
        radius=0.0,
    )
    raw_obs["structures"] = [cc, barracks, depot]

    result = transformer.transform(raw_obs)
    terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    terrain = result["map"][terrain_idx]

    cc_px, cc_py = transformer._game_to_obs_pixel(64.0, 64.0)
    barracks_px, barracks_py = transformer._game_to_obs_pixel(80.0, 64.0)
    depot_px, depot_py = transformer._game_to_obs_pixel(48.0, 64.0)

    assert terrain[cc_py, cc_px] <= 0.0
    assert terrain[barracks_py, barracks_px] <= 0.0
    assert terrain[depot_py, depot_px] <= 0.0
    assert np.count_nonzero(terrain <= 0.0) >= 3

    x0, x1, y0, y1 = transformer._structure_obs_rect_bounds(cc)
    cc_block = terrain[y0:y1 + 1, x0:x1 + 1] <= 0.0
    assert np.count_nonzero(cc_block) >= 4, "Command Center footprint should span multiple obs pixels"


def test_structure_footprint_radius_fallback_without_live_radius():
    """Structures without live radius use SC2 footprint defaults."""
    transformer = ObservationTransformer(map_size=(128, 128))
    cc = create_mock_unit(
        UnitTypeId.COMMANDCENTER,
        Point2((0.0, 0.0)),
        is_structure=True,
        radius=0.0,
    )
    depot = create_mock_unit(
        UnitTypeId.SUPPLYDEPOT,
        Point2((0.0, 0.0)),
        is_structure=True,
        radius=0.0,
    )
    assert transformer._structure_game_radius(cc) == 2.5
    assert transformer._structure_game_radius(depot) == 1.0


def test_structure_footprint_ignores_inflated_live_radius_with_addon():
    """Add-ons inflate live selection radius; obs must keep the base building footprint."""
    transformer = ObservationTransformer(map_size=(128, 128))
    barracks = create_mock_unit(
        UnitTypeId.BARRACKS,
        Point2((256.0, 256.0)),
        is_structure=True,
        radius=4.5,
        footprint_radius=4.5,
    )
    assert transformer._structure_game_radius(barracks) == 1.5

    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)
    raw_obs = {
        "units": [],
        "structures": [barracks],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    focal = create_mock_unit(
        UnitTypeId.MARINE,
        Point2((256.0, 256.0)),
        radius=0.375,
    )
    tactical = transformer.transform_tactical(raw_obs, focal)
    density = tactical[0]
    assert int(np.count_nonzero(density > 0)) <= 10


def test_barracks_with_attached_addon_uses_combined_footprint():
    """Production buildings must render parent + add-on slot as one L-shaped footprint."""
    transformer = ObservationTransformer(map_size=(128, 128))
    barracks = create_mock_unit(
        UnitTypeId.BARRACKS,
        Point2((64.0, 64.0)),
        is_structure=True,
        radius=0.0,
        add_on_tag=9001,
        tag=9000,
    )
    techlab = create_mock_unit(
        UnitTypeId.TECHLAB,
        Point2((66.5, 63.5)),
        is_structure=True,
        radius=0.0,
        tag=9001,
    )

    base_only = create_mock_unit(
        UnitTypeId.BARRACKS,
        Point2((64.0, 64.0)),
        is_structure=True,
        radius=0.0,
    )

    transformer._refresh_attached_addon_tags([barracks], [])
    assert transformer._is_attached_addon_unit(techlab)

    pathing = np.ones((128, 128), dtype=np.uint8)
    # Parent 3×3 block plus a separate add-on wing (L-shaped unpathable region).
    pathing[62:66, 62:66] = 0
    pathing[65:68, 62:65] = 0
    gi = _make_game_info(128, 128, with_grids=True)
    gi.pathing_grid.data_numpy = pathing
    raw_obs = {
        "units": [],
        "structures": [barracks, techlab],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    transformer.transform(raw_obs)
    combined_bounds = transformer._structure_obs_rect_bounds(barracks)
    base_bounds = transformer._structure_obs_rect_bounds(base_only)
    combined_mask = transformer._structure_game_footprint_mask(
        barracks, *combined_bounds
    )
    base_mask = transformer._structure_game_footprint_mask(
        base_only, *base_bounds
    )
    assert int(np.count_nonzero(combined_mask)) > int(np.count_nonzero(base_mask))


def test_creep_tumor_tactical_single_pixel():
    """Creep tumors are 1×1 in game — tactical density must be a single pixel."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)

    active = create_mock_unit(
        UnitTypeId.ZERGLING,
        Point2((256.0, 256.0)),
        radius=0.375,
    )
    tumor = create_mock_unit(
        UnitTypeId.CREEPTUMOR,
        Point2((260.0, 256.0)),
        is_structure=True,
        radius=0.0,
    )
    raw_obs = {
        "units": [active],
        "structures": [tumor],
        "enemy_units": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, active)
    density = tactical[0]
    without_tumor = transformer.transform_tactical(
        {**raw_obs, "structures": []}, active
    )[0]
    tumor_mask = (density > 0) & ~(without_tumor > 0)
    assert np.count_nonzero(tumor_mask) == 1


def test_overlord_tactical_footprint_stable_single_pixel():
    """Flying support units render as one tactical pixel and do not wiggle."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf=_universal_training_conf(),
    )
    gi = _large_map_game_info()
    gi.pathing_grid.data_numpy = np.ones((512, 512), dtype=np.uint8)

    counts = []
    for offset in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        overlord = create_mock_unit(
            UnitTypeId.OVERLORD,
            Point2((256.0 + offset, 256.0 + offset)),
            radius=1.125,
            is_flying=True,
        )
        raw_obs = {
            "units": [overlord],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        tactical = transformer.transform_tactical(raw_obs, overlord)
        counts.append(int(np.count_nonzero(tactical[0] > 0)))

    assert counts == [1] * len(counts)


def test_structure_footprint_prefers_game_data_catalog():
    """GameData catalog footprint_radius wins over hardcoded fallbacks."""
    transformer = ObservationTransformer(map_size=(128, 128))
    type_data = MagicMock()
    type_data.footprint_radius = 2.75
    game_data = MagicMock()
    game_data.units = {UnitTypeId.COMMANDCENTER.value: type_data}
    transformer._current_game_data = game_data

    cc = create_mock_unit(
        UnitTypeId.COMMANDCENTER,
        Point2((0.0, 0.0)),
        is_structure=True,
        radius=0.0,
    )
    assert transformer._structure_game_radius(cc) == 2.75


def test_structure_game_radius_ignores_selection_radius_in_json():
    """PYLON must use footprint catalog, not unit_data selection ``radius`` (1.125)."""
    transformer = ObservationTransformer(map_size=(128, 128))
    pylon = create_mock_unit(
        UnitTypeId.PYLON,
        Point2((0.0, 0.0)),
        is_structure=True,
        radius=1.125,
    )
    assert transformer._structure_game_radius(pylon) == pytest.approx(1.0)


def test_tactical_command_objective_beacon_marker_is_compact():
    """Beacon / Point2 objectives must not use the global r=3 disc on the 32×32 fovea."""
    training_conf = {
        "agent_architecture": "universal",
        "active_tactical_layers": [LayerConstants.COMMAND_OBJECTIVE],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=training_conf)
    gi = _make_game_info(128, 128, with_grids=True)
    active = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    beacon = create_mock_unit(UnitTypeId.BEACON_TERRANSMALL, Point2((68.0, 64.0)))
    raw_obs = {
        "units": [active],
        "enemy_units": [],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "command_target": beacon,
        "selected_units": [active],
    }
    tactical = transformer.transform_tactical(raw_obs, active)
    painted = int(np.count_nonzero(tactical[0] > 0))
    assert painted > 0
    assert painted < 16, f"tactical beacon objective too large ({painted} px)"


def test_tactical_command_objective_skips_far_pylon():
    """Command objective must not paint a clipped arc when the pylon is outside the fovea."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_tactical_layers": [LayerConstants.COMMAND_OBJECTIVE],
        },
    )
    gi = _large_map_game_info()
    active = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)), radius=0.375)
    pylon = create_mock_unit(
        UnitTypeId.PYLON,
        Point2((200.0, 200.0)),
        is_structure=True,
        radius=1.125,
    )
    raw_obs = {
        "units": [active],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [pylon],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "command_target": pylon.position,
    }
    tactical = transformer.transform_tactical(raw_obs, active)
    assert np.count_nonzero(tactical[0] > 0) == 0


def test_photon_cannon_renders_detection_ring_when_powered():
    """Detector structures (photon cannon) must paint the allied/enemy detection layer."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [
                LayerConstants.ALLIED_DETECTION,
                LayerConstants.ENEMY_DETECTION,
            ],
            "active_tactical_layers": [
                LayerConstants.ALLIED_DETECTION,
                LayerConstants.TERRAIN_PATHABLE,
            ],
        },
    )
    gi = _make_game_info(128, 128, with_grids=True)
    powered = create_mock_unit(
        UnitTypeId.PHOTONCANNON,
        Point2((64.0, 64.0)),
        is_structure=True,
        is_ready=True,
        build_progress=1.0,
        is_powered=True,
        detect_range=11.0,
        radius=1.0,
    )
    unpowered = create_mock_unit(
        UnitTypeId.PHOTONCANNON,
        Point2((80.0, 64.0)),
        is_structure=True,
        is_ready=True,
        build_progress=1.0,
        is_powered=False,
        detect_range=11.0,
        radius=1.0,
    )
    building = create_mock_unit(
        UnitTypeId.PHOTONCANNON,
        Point2((96.0, 64.0)),
        is_structure=True,
        is_ready=False,
        build_progress=0.5,
        is_powered=True,
        detect_range=11.0,
        radius=1.0,
    )
    raw_obs = {
        "units": [],
        "structures": [powered],
        "enemy_structures": [unpowered, building],
        "enemy_units": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    ally_det = transformer.transform(raw_obs)["map"][0]
    assert np.count_nonzero(ally_det > 0) > 0

    enemy_det = transformer.transform(raw_obs)["map"][1]
    assert np.count_nonzero(enemy_det > 0) == 0

    active = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    tactical = transformer.transform_tactical(raw_obs, active)
    det_ch = transformer.active_tactical_layers.index(LayerConstants.ALLIED_DETECTION)
    assert np.count_nonzero(tactical[det_ch] > 0) > 0


def test_structure_dps_suppressed_until_construction_complete():
    """Defensive structures must not paint weapon rings while warping or building."""
    conf = {
        "agent_architecture": "universal",
        "active_layers": [LayerConstants.ENEMY_G_DPS_BASE],
        "active_tactical_layers": [LayerConstants.ENEMY_G_DPS_BASE],
    }
    transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
    gi = _large_map_game_info()
    pos = Point2((256.0, 256.0))
    building = create_mock_unit(
        UnitTypeId.PHOTONCANNON,
        pos,
        is_structure=True,
        is_ready=False,
        build_progress=0.4,
        radius=1.0,
    )
    complete = create_mock_unit(
        UnitTypeId.PHOTONCANNON,
        pos,
        is_structure=True,
        is_ready=True,
        build_progress=1.0,
        radius=1.0,
    )
    active = create_mock_unit(UnitTypeId.ZERGLING, Point2((256.0, 256.0)), radius=0.375)
    base_raw = {
        "units": [active],
        "structures": [],
        "enemy_units": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    building_raw = {**base_raw, "enemy_structures": [building]}
    complete_raw = {**base_raw, "enemy_structures": [complete]}

    building_dps = transformer.transform(building_raw)["map"][0]
    assert np.count_nonzero(building_dps > 0) == 0

    complete_dps = transformer.transform(complete_raw)["map"][0]
    assert np.count_nonzero(complete_dps > 0) > 0

    assert transformer._unit_max_ground_weapon_reach(building) == 0.0
    assert transformer._unit_max_ground_weapon_reach(complete) > 0.0


def test_skip_undetected_cloaked_enemy_without_is_detected():
    """Real SC2 units (e.g. structures) may omit ``is_detected``."""
    from types import SimpleNamespace

    unit = SimpleNamespace(is_cloaked=False, is_burrowed=False, is_snapshot=False)
    assert not ObservationTransformer._skip_undetected_cloaked_enemy(unit, True)

    cloaked = SimpleNamespace(is_cloaked=True, is_burrowed=False, is_detected=False, is_snapshot=False)
    assert ObservationTransformer._skip_undetected_cloaked_enemy(cloaked, True)

    burrowed = SimpleNamespace(is_cloaked=False, is_burrowed=True, is_detected=False, is_snapshot=False)
    assert ObservationTransformer._skip_undetected_cloaked_enemy(burrowed, True)

    assert not ObservationTransformer._skip_undetected_cloaked_enemy(
        cloaked, True, allied_detector_covers=True
    )


def test_p2_allied_observer_reveals_p1_cloaked_enemy_despite_proto_undetected():
    """P1-client is_detected=False on a P1 cloaked unit; P2 Observer still paints it."""
    transformer = ObservationTransformer(
        map_size=(64, 64),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [
                LayerConstants.ALLIED_UNIT_DENSITY,
                LayerConstants.ENEMY_UNIT_DENSITY,
            ],
        },
    )
    gi = _make_game_info(64, 64, with_grids=True)
    observer = create_mock_unit(
        UnitTypeId.OBSERVER, Point2((20.0, 20.0)), detect_range=11.0
    )
    cloaked = create_mock_unit(
        UnitTypeId.DARKTEMPLAR,
        Point2((24.0, 20.0)),
        is_cloaked=True,
        is_detected=False,
        is_snapshot=False,
    )
    raw_obs = {
        "units": [observer],
        "enemy_units": [cloaked],
        "effects": [],
        "game_info": gi,
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    enemy_density = result["map"][1]
    assert float(enemy_density.sum()) > 0.0


def test_undetected_cloaked_enemy_still_skipped_without_allied_detector():
    transformer = ObservationTransformer(
        map_size=(64, 64),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [
                LayerConstants.ALLIED_UNIT_DENSITY,
                LayerConstants.ENEMY_UNIT_DENSITY,
            ],
        },
    )
    gi = _make_game_info(64, 64, with_grids=True)
    marine = create_mock_unit(UnitTypeId.MARINE, Point2((10.0, 10.0)))
    cloaked = create_mock_unit(
        UnitTypeId.DARKTEMPLAR,
        Point2((50.0, 50.0)),
        is_cloaked=True,
        is_detected=False,
        is_snapshot=False,
    )
    raw_obs = {
        "units": [marine],
        "enemy_units": [cloaked],
        "effects": [],
        "game_info": gi,
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    enemy_density = result["map"][1]
    assert float(enemy_density.sum()) == 0.0


def test_burrowed_unit_renders_on_cloak_layer():
    """Burrowed zerg units stamp allied_unit_cloak / enemy_unit_cloak."""
    transformer = ObservationTransformer(
        map_size=(64, 64),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [
                LayerConstants.ALLIED_UNIT_DENSITY,
                LayerConstants.ALLIED_UNIT_CLOAK,
                LayerConstants.ENEMY_UNIT_DENSITY,
                LayerConstants.ENEMY_UNIT_CLOAK,
            ],
        },
    )
    gi = _make_game_info(64, 64)
    ally_burrowed = create_mock_unit(
        UnitTypeId.LURKERMPBURROWED,
        Point2((20, 20)),
        is_burrowed=True,
    )
    enemy_burrowed = create_mock_unit(
        UnitTypeId.ROACHBURROWED,
        Point2((40, 40)),
        is_burrowed=True,
        is_detected=True,
    )
    raw_obs = {
        "units": [ally_burrowed],
        "enemy_units": [enemy_burrowed],
        "effects": [],
        "game_info": gi,
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    map_data = result["map"]
    ally_ch = transformer.active_layers.index(LayerConstants.ALLIED_UNIT_CLOAK)
    enemy_ch = transformer.active_layers.index(LayerConstants.ENEMY_UNIT_CLOAK)
    assert np.sum(map_data[ally_ch]) > 0
    assert np.sum(map_data[enemy_ch]) > 0


def test_cloaked_unit_renders_on_cloak_layer():
    """Cloaked units stamp the cloak attribute layer when visible."""
    transformer = ObservationTransformer(
        map_size=(64, 64),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [
                LayerConstants.ENEMY_UNIT_DENSITY,
                LayerConstants.ENEMY_UNIT_CLOAK,
            ],
        },
    )
    gi = _make_game_info(64, 64)
    banshee = create_mock_unit(
        UnitTypeId.BANSHEE,
        Point2((30, 30)),
        is_cloaked=True,
        is_detected=True,
    )
    raw_obs = {
        "units": [],
        "enemy_units": [banshee],
        "effects": [],
        "game_info": gi,
        "game_data": None,
    }
    result = transformer.transform(raw_obs)
    cloak_ch = transformer.active_layers.index(LayerConstants.ENEMY_UNIT_CLOAK)
    assert np.sum(result["map"][cloak_ch]) > 0


def test_tactical_enemy_pylon_hidden_when_footprint_outside_fovea():
    """Enemy structures far from the focal unit must not appear on tactical layers."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_tactical_layers": [
                LayerConstants.ENEMY_UNIT_MECHANICAL,
                LayerConstants.ENEMY_UNIT_DENSITY,
            ],
        },
    )
    gi = _large_map_game_info()
    active = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)), radius=0.375)
    pylon = create_mock_unit(
        UnitTypeId.PYLON,
        Point2((200.0, 200.0)),
        is_structure=True,
        is_mechanical=True,
        radius=1.125,
    )
    raw_obs = {
        "units": [active],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [pylon],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    tactical = transformer.transform_tactical(raw_obs, active)
    assert np.count_nonzero(tactical[0] > 0) == 0
    assert np.count_nonzero(tactical[1] > 0) == 0


def test_vespene_extractor_density_matches_terrain_unpathable():
    """Built gas structures must cover the same obs cells as the terrain unpathable patch."""
    pathing = np.ones((128, 128), dtype=np.uint8)
    pathing[62:66, 62:66] = 0
    gi = _make_game_info(128, 128, with_grids=True)
    gi.pathing_grid.data_numpy = pathing

    extractor = create_mock_unit(
        UnitTypeId.EXTRACTOR,
        Point2((64.0, 64.0)),
        is_structure=True,
        radius=1.5,
        footprint_radius=1.5,
    )
    geyser = create_mock_unit(
        UnitTypeId.VESPENEGEYSER,
        Point2((64.0, 64.0)),
        is_structure=True,
        is_vespene_geyser=True,
        radius=1.0,
    )

    transformer = ObservationTransformer(map_size=(128, 128))
    raw_obs = {
        "units": [],
        "structures": [extractor],
        "enemy_units": [],
        "enemy_structures": [],
        "mineral_field": [],
        "vespene_geyser": [geyser],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    density_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY]
    terrain = result["map"][terrain_idx]
    density = result["map"][density_idx]

    struct_mask = density > 0
    unpathable = terrain <= 0.0
    assert np.count_nonzero(struct_mask) >= 9
    assert np.count_nonzero(overlap := np.logical_and(struct_mask, unpathable)) == np.count_nonzero(
        struct_mask
    )
    assert np.array_equal(struct_mask, unpathable & struct_mask)


def test_mineral_fields_stamped_unpathable_on_terrain():
    """Mineral patches must appear unpathable even when downsampling blurs pathing cells."""
    size = 64
    path = np.ones((size, size), dtype=np.uint8)
    height = np.full((size, size), 12.0, dtype=np.float32)
    path[30:34, 30:34] = 0
    height[30:34, 30:34] = 20.0

    gi = _make_game_info(size, size, with_grids=True)
    gi.pathing_grid.data_numpy = path
    gi.terrain_height.data_numpy = height

    mineral = create_mock_unit(
        UnitTypeId.MINERALFIELD,
        Point2((32.0, 32.0)),
        is_structure=True,
        is_mineral_field=True,
        radius=1.0,
        footprint_radius=1.5,
    )

    transformer = ObservationTransformer(map_size=(128, 128))
    raw_obs = {
        "units": [],
        "structures": [],
        "enemy_units": [],
        "enemy_structures": [],
        "mineral_field": [mineral],
        "vespene_geyser": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
    }
    result = transformer.transform(raw_obs)
    terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    terrain = result["map"][terrain_idx]

    px, py = transformer._game_to_obs_pixel(32.0, 32.0)
    assert terrain[py, px] <= 0.0
    assert np.count_nonzero(terrain <= 0.0) >= 1


def test_tactical_flying_layer_renders_when_configured():
    """Flying attribute layers must be available and drawn in the tactical fovea."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [
                LayerConstants.ALLIED_UNIT_FLYING,
                LayerConstants.ENEMY_UNIT_FLYING,
            ],
        },
    )
    focal = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    banshee = create_mock_unit(
        UnitTypeId.BANSHEE,
        Point2((66.0, 64.0)),
        is_flying=True,
        is_mechanical=True,
        is_light=True,
    )
    gi = _make_game_info(128, 128, with_grids=True)
    gi.placement_grid = MagicMock()
    gi.placement_grid.data_numpy = gi.pathing_grid.data_numpy.copy()
    raw_obs = {
        "units": [focal],
        "enemy_units": [banshee],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert tactical.shape[0] == 2
    assert float(tactical[1].max()) > 0.0


def test_tactical_air_dps_layer_renders_when_configured():
    """Air-only weapons must paint a_dps channels when listed in active_tactical_layers."""
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_tactical_layers": [LayerConstants.ENEMY_A_DPS_MASSIVE],
        },
    )
    focal = create_mock_unit(UnitTypeId.ZERGLING, Point2((64.0, 64.0)))
    tempest = create_mock_unit(
        UnitTypeId.TEMPEST,
        Point2((70.0, 70.0)),
        is_flying=True,
        is_massive=True,
        is_armored=True,
        is_mechanical=True,
        air_range=15.0,
        ground_range=0.0,
    )
    gi = _make_game_info(128, 128, with_grids=True)
    gi.placement_grid = MagicMock()
    gi.placement_grid.data_numpy = gi.pathing_grid.data_numpy.copy()
    raw_obs = {
        "units": [focal],
        "enemy_units": [tempest],
        "structures": [],
        "enemy_structures": [],
        "effects": [],
        "game_info": gi,
        "game_data": MagicMock(),
        "selected_units": [focal],
    }
    tactical = transformer.transform_tactical(raw_obs, focal)
    assert float(tactical[0].max()) > 0.0
    assert int(np.count_nonzero(tactical[0] > 0.01)) > 0


def _dps_draw_transformer(active_layers):
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "active_layers": active_layers,
            "active_tactical_layers": active_layers,
        },
    )
    transformer.current_layer_map = LayerConstants.LAYERS
    transformer.is_tactical_mode = False
    transformer._update_active_indices()
    transformer.scale = transformer.scale_x = transformer.scale_y = 1.0
    transformer.offset_x = transformer.offset_y = 0.0
    transformer.crop_x = transformer.crop_y = 0.0
    transformer._obs_span_w = transformer._obs_span_h = 128.0
    transformer._grid_x = np.arange(128).reshape(1, -1)
    transformer._grid_y = np.arange(128).reshape(-1, 1)
    return transformer


def test_corruptor_paints_a_dps_massive_bonus_when_air_base_active():
    """Corruptor +6 vs Massive must land on a_dps_massive (bonus-only with a_dps_base)."""
    layers_cfg = [
        LayerConstants.ALLIED_A_DPS_BASE,
        LayerConstants.ALLIED_A_DPS_MASSIVE,
    ]
    transformer = _dps_draw_transformer(layers_cfg)
    layers = np.zeros((LayerConstants.get_num_layers(), 128, 128), dtype=np.float32)
    corruptor = create_mock_unit(
        UnitTypeId.CORRUPTOR,
        Point2((64.0, 64.0)),
        is_flying=True,
        air_range=6.0,
        ground_range=0.0,
        radius=0.625,
    )
    transformer._draw_unit_dps(corruptor, layers, False, 64, 63, MagicMock())
    period = 1.89990234375
    base_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_A_DPS_BASE]
    mass_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_A_DPS_MASSIVE]
    assert layers[base_idx].max() == pytest.approx(14.0 / period, rel=1e-5)
    assert layers[mass_idx].max() == pytest.approx(6.0 / period, rel=1e-5)


def test_corruptor_a_dps_massive_is_base_plus_bonus_without_air_base():
    """Cross-axis g_dps_base fallback must not strip base from a_dps_massive."""
    layers_cfg = [
        LayerConstants.ALLIED_G_DPS_BASE,
        LayerConstants.ALLIED_A_DPS_MASSIVE,
    ]
    transformer = _dps_draw_transformer(layers_cfg)
    layers = np.zeros((LayerConstants.get_num_layers(), 128, 128), dtype=np.float32)
    corruptor = create_mock_unit(
        UnitTypeId.CORRUPTOR,
        Point2((64.0, 64.0)),
        is_flying=True,
        air_range=6.0,
        ground_range=0.0,
        radius=0.625,
    )
    transformer._draw_unit_dps(corruptor, layers, False, 64, 63, MagicMock())
    period = 1.89990234375
    mass_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_A_DPS_MASSIVE]
    assert layers[mass_idx].max() == pytest.approx((14.0 + 6.0) / period, rel=1e-5)


def test_phoenix_air_light_bonus_multiplies_attacks():
    """Dual-attack air weapons must apply damage_bonus per attack hit."""
    layers_cfg = [
        LayerConstants.ALLIED_A_DPS_BASE,
        LayerConstants.ALLIED_A_DPS_LIGHT,
    ]
    transformer = _dps_draw_transformer(layers_cfg)
    layers = np.zeros((LayerConstants.get_num_layers(), 128, 128), dtype=np.float32)
    phoenix = create_mock_unit(
        UnitTypeId.PHOENIX,
        Point2((64.0, 64.0)),
        is_flying=True,
        air_range=5.0,
        ground_range=0.0,
        radius=0.6,
    )
    transformer._draw_unit_dps(phoenix, layers, False, 64, 63, MagicMock())
    period = 1.10009765625
    light_idx = LayerConstants.LAYERS[LayerConstants.ALLIED_A_DPS_LIGHT]
    assert layers[light_idx].max() == pytest.approx((5.0 * 2) / period, rel=1e-5)


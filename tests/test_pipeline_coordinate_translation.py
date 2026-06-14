"""
Integration tests: global + tactical coordinate translation on every pipeline map.

Reads real SC2Map archives (MapInfo + terrain metadata) and verifies:
- All 1024 global spatial cells decode into the same obs cell as ObservationTransformer.
- All 1024 tactical (move_local) cells match world_to_tactical_obs_pixel.
- Every integer world coordinate in playable_area round-trips to the same spatial cell
  (exhaustive on small maps; spatial-cell centers + edge anchors on large ladder maps).

Requires mpyq (same dependency as CTF region tests).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sc2.position import Point2

from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
from src.transformers.observation_transformer import ObservationTransformer
from src.utils.obs_geometry import world_to_tactical_obs_pixel
from tests.map_geometry_loader import (
    MapGeometry,
    _MPYQ_AVAILABLE,
    build_mock_game_info,
    collect_pipeline_map_geometries,
    iter_fovea_playable_integers,
    roundtrip_world_sample_points,
    sync_transformer_global_geometry,
)

pytestmark = pytest.mark.skipif(
    not _MPYQ_AVAILABLE,
    reason="mpyq not installed — install with 'pip install mpyq' to run pipeline map tests.",
)

SPATIAL_RES = 32
GLOBAL_STEP = 128 // SPATIAL_RES

if _MPYQ_AVAILABLE:
    try:
        _PIPELINE_MAP_CASES = collect_pipeline_map_geometries("config_universal")
        _PIPELINE_MAP_IDS = [
            f"{level}::{geo.map_name}" for level, geo in _PIPELINE_MAP_CASES
        ]
    except Exception:
        _PIPELINE_MAP_CASES = []
        _PIPELINE_MAP_IDS = []
else:
    _PIPELINE_MAP_CASES = []
    _PIPELINE_MAP_IDS = []


def _focal_unit(geometry: MapGeometry) -> MagicMock:
    unit = MagicMock()
    unit.position = Point2(
        (
            geometry.playable_x + geometry.playable_width / 2.0,
            geometry.playable_y + geometry.playable_height / 2.0,
        )
    )
    return unit


def _make_agent_and_transformer(geometry: MapGeometry):
    game_info, path = build_mock_game_info(geometry)
    transformer = ObservationTransformer(
        map_size=(128, 128),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": ["terrain_pathable"],
            "active_tactical_layers": ["terrain_pathable"],
        },
    )
    sync_transformer_global_geometry(transformer, game_info, path)
    agent = UniversalUnitControlAgent(cnn_input_size=(128, 128), grid_size=32, dilation_size=0)
    agent.last_game_info = game_info
    return agent, transformer, _focal_unit(geometry)


def _pipeline_maps() -> list[tuple[str, MapGeometry]]:
    if not _PIPELINE_MAP_CASES:
        pytest.skip("No SC2Map files found for config_universal pipeline levels.")
    return _PIPELINE_MAP_CASES


def test_pipeline_map_catalog_non_empty():
    maps = _pipeline_maps()
    assert len(maps) >= 10
    stems = {g.map_name for _, g in maps}
    assert "00_32x32_No_Middle" in stems
    assert any("Persephone" in s or "Interloper" in s for s in stems)


@pytest.mark.parametrize(
    "level_name,map_geometry",
    _PIPELINE_MAP_CASES,
    ids=_PIPELINE_MAP_IDS,
)
def test_global_spatial_cells_match_observation_transformer(level_name, map_geometry):
    """Decode must use the same geometry as ObservationTransformer (pixel-for-pixel)."""
    agent, transformer, focal = _make_agent_and_transformer(map_geometry)
    for idx in range(SPATIAL_RES * SPATIAL_RES):
        wx, wy = agent.convert_spatial_action_to_world(
            "move_global", idx, active_unit=focal
        )
        tx, ty = transformer._game_to_obs_pixel(wx, wy)
        px, py = agent.world_to_cnn_pixel(wx, wy)
        assert (px, py) == (tx, ty), (
            f"{map_geometry.map_name} ({level_name}): idx={idx} agent/obs pixel mismatch "
            f"agent=({px},{py}) obs=({tx},{ty})"
        )


@pytest.mark.parametrize(
    "level_name,map_geometry",
    _PIPELINE_MAP_CASES,
    ids=_PIPELINE_MAP_IDS,
)
def test_tactical_spatial_cells_match_observation_transformer(level_name, map_geometry):
    agent, _transformer, focal = _make_agent_and_transformer(map_geometry)
    fx = float(focal.position.x)
    fy = float(focal.position.y)
    for head in ("move_local", "attack_local"):
        for idx in range(SPATIAL_RES * SPATIAL_RES):
            wx, wy = agent.convert_spatial_action_to_world(
                head, idx, active_unit=focal
            )
            px, py = world_to_tactical_obs_pixel(wx, wy, fx, fy)
            gy, gx = divmod(idx, SPATIAL_RES)
            assert (px, py) == (gx, gy), (
                f"{map_geometry.map_name} ({level_name}): {head} idx={idx} "
                f"tactical ({px},{py}) != ({gx},{gy})"
            )


@pytest.mark.parametrize(
    "level_name,map_geometry",
    _PIPELINE_MAP_CASES,
    ids=_PIPELINE_MAP_IDS,
)
def test_all_playable_integer_positions_share_spatial_cell(level_name, map_geometry):
    """Playable world points must encode/decode within one spatial cell."""
    agent, transformer, focal = _make_agent_and_transformer(map_geometry)
    path = agent.last_game_info.pathing_grid.data_numpy
    game_info = agent.last_game_info
    for x, y in roundtrip_world_sample_points(map_geometry, game_info, path):
        pos = MagicMock()
        pos.x = x
        pos.y = y
        idx = agent.get_grid_index_at_world_pos(pos)
        dx, dy = agent.convert_spatial_action_to_world(
            "move_global", idx, active_unit=focal
        )
        px0, py0 = transformer._game_to_obs_pixel(x, y)
        px1, py1 = transformer._game_to_obs_pixel(dx, dy)
        assert px0 // GLOBAL_STEP == px1 // GLOBAL_STEP, (
            f"{map_geometry.map_name} ({level_name}): world ({x},{y}) "
            f"cell x mismatch idx={idx} decode=({dx:.2f},{dy:.2f})"
        )
        assert py0 // GLOBAL_STEP == py1 // GLOBAL_STEP, (
            f"{map_geometry.map_name} ({level_name}): world ({x},{y}) "
            f"cell y mismatch idx={idx} decode=({dx:.2f},{dy:.2f})"
        )


@pytest.mark.parametrize(
    "level_name,map_geometry",
    _PIPELINE_MAP_CASES,
    ids=_PIPELINE_MAP_IDS,
)
def test_all_playable_integer_positions_tactical_cell(level_name, map_geometry):
    """Playable coords in the tactical fovea round-trip move_local cells."""
    agent, _transformer, focal = _make_agent_and_transformer(map_geometry)
    fx = float(focal.position.x)
    fy = float(focal.position.y)
    for x, y in iter_fovea_playable_integers(map_geometry, fx, fy):
        px, py = world_to_tactical_obs_pixel(float(x), float(y), fx, fy)
        idx = py * SPATIAL_RES + px
        dx, dy = agent.convert_spatial_action_to_world(
            "move_local", idx, active_unit=focal
        )
        px1, py1 = world_to_tactical_obs_pixel(dx, dy, fx, fy)
        assert (px1, py1) == (px, py), (
            f"{map_geometry.map_name} ({level_name}): tactical roundtrip "
            f"({x},{y}) idx={idx} -> ({dx:.2f},{dy:.2f}) -> ({px1},{py1})"
        )

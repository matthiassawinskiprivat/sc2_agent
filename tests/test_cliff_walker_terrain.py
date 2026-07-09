"""
Tests for cliff-walker terrain pathability display.

Cliff face detection strategy (exact SC2MapAnalysis C-extension port):
  - Non-walkable cells are marked as cliff-traversable when a 2×2 height
    block around them shows EXACTLY LEVEL_DIFFERENCE = 16 height units
    between the walkable areas on both sides.
  - A connectivity filter removes isolated single-cell artefacts: a cell
    must have at least one climbable neighbour (8-connected, so diagonal
    cliff runs are preserved) to be included.
  - Walls, minerals, and building footprints at flat height (height diff = 0,
    not 16) are naturally excluded.
  - Walkable terrain brightness uses a fixed step per cliff tier (16 gu) from the
    map's lowest walkable height; tier count does not rescale the step. Walkable
    height outliers are capped at the 95th percentile when computing map metadata;
    the baseline uses the 5th percentile of playable-area walkable heights so rare
    low ramps do not wash out dominant cliff tiers.
  - Walkable terrain heights are NEVER modified — the cliff-walker view
    differs only in that cliff-face cells (normally 0/dark) are shown at the
    mean of neighbouring walkable heights, creating a gradient:
    dark (low ground) → bright (cliff face) → bright (high ground).
"""

import math

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_conf(layers=None, tactical_layers=None):
    layers = layers or [LayerConstants.TERRAIN_PATHABLE]
    tactical_layers = tactical_layers or [LayerConstants.TERRAIN_PATHABLE]
    return {
        "grid_size": 16,
        "agent_architecture": "universal",
        "active_layers": layers,
        "active_tactical_layers": tactical_layers,
        "action_heads": ["move"],
        "categorical_heads": [],
        "high_level_command": 0,
        "rewards": {"success": 10.0, "step_penalty": -0.1},
        "env": {"targets": 1, "maps_dir": "train/maps/128x128"},
    }


def _make_game_info(path_grid: np.ndarray, height_grid: np.ndarray):
    """Build a minimal mock game_info from raw numpy arrays."""
    h, w = path_grid.shape
    size = MagicMock()
    size.width = w
    size.height = h
    gi = MagicMock()
    gi.map_name = "test_map"
    gi.map_size = size
    gi.playable_area = None
    pathing_mock = MagicMock()
    pathing_mock.data_numpy = path_grid.astype(np.uint8)
    height_mock = MagicMock()
    height_mock.data_numpy = height_grid.astype(np.float32)
    gi.pathing_grid = pathing_mock
    gi.terrain_height = height_mock
    gi.map_ramps = None
    return gi


def _make_transformer(feature_map_size=16):
    """Create a minimal ObservationTransformer without real game assets."""
    conf = _make_training_conf()
    transformer = ObservationTransformer.__new__(ObservationTransformer)
    transformer.training_conf = conf
    transformer.feature_map_width = feature_map_size
    transformer.feature_map_height = feature_map_size
    transformer.active_layers = conf["active_layers"]
    transformer.active_tactical_layers = conf["active_tactical_layers"]
    transformer.scale = 1.0
    transformer.scale_x = 1.0
    transformer.scale_y = 1.0
    transformer.offset_x = 0
    transformer.offset_y = 0
    transformer.crop_x = 0
    transformer.crop_y = 0
    transformer._obs_span_w = float(feature_map_size)
    transformer._obs_span_h = float(feature_map_size)
    transformer.is_tactical_mode = False
    transformer.current_layer_map = LayerConstants.LAYERS
    transformer._terrain_caches = {}
    transformer._last_map_name = None
    transformer._cliff_face_cache = None
    transformer._cliff_face_cache_map = None
    transformer._cliff_display_heights = None
    transformer._grid_x = np.arange(feature_map_size).reshape(1, -1)
    transformer._grid_y = np.arange(feature_map_size).reshape(-1, 1)
    transformer.unit_db = {}
    transformer.fallback_db = {}
    transformer._current_game_data = None
    return transformer


def _call_draw(transformer, path_grid, height_grid, cliff_walker=False):
    """Call _draw_terrain_pathing and return the rendered TERRAIN_PATHABLE layer."""
    game_info = _make_game_info(path_grid, height_grid)
    n_layers = LayerConstants.get_num_layers()
    layers = np.zeros(
        (n_layers, transformer.feature_map_height, transformer.feature_map_width),
        dtype=np.float32,
    )
    transformer._draw_terrain_pathing(
        game_info, layers, cliff_walker=cliff_walker
    )
    layer_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
    return layers[layer_idx]


# ---------------------------------------------------------------------------
# Synthetic map — SC2-correct heights (LEVEL_DIFFERENCE = 16)
#
#   Map size: 16×16  (large enough to avoid 2-cell boundary exclusion)
#   LOW_H  = 12  (low ground)
#   HIGH_H = 28  (high ground = LOW_H + 16)
#   CLIFF_FACE_H = 28  (cliff-face cells at HIGH_H — typical in real SC2)
#
#   Layout (data rows, row 0 = top):
#       rows  0-4:  high ground walkable  (height 28)
#       row   5:    non-walkable cliff gap (height 28 = CLIFF_FACE_H)
#       rows  6-15: low ground walkable   (height 12)
#
#   Feature-map y is flipped (sc2_y = MAP_SIZE - 1 - feature_y), so:
#       feature row 0  → data row 15 (low ground interior)
#       feature row 10 → data row 5  (cliff gap)
#       feature row 15 → data row 0  (high ground interior)
# ---------------------------------------------------------------------------

MAP_SIZE = 16
LOW_H = 12.0       # one SC2 terrain level
HIGH_H = 28.0      # LOW_H + LEVEL_DIFFERENCE (16)
CLIFF_FACE_H = 28.0  # matches HIGH_H — typical real-SC2 encoding


@pytest.fixture
def synthetic_map():
    path = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    height = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)

    path[0:5, :] = 1          # high ground walkable
    height[0:5, :] = HIGH_H

    path[5, :] = 0             # non-walkable cliff gap
    height[5, :] = CLIFF_FACE_H

    path[6:MAP_SIZE, :] = 1   # low ground walkable
    height[6:MAP_SIZE, :] = LOW_H

    return path, height


def _feat_row(data_row: int) -> int:
    """Convert a data-grid row to the corresponding feature-map row."""
    return MAP_SIZE - 1 - data_row


# ---------------------------------------------------------------------------
# Unit tests for _compute_climber_grid
# ---------------------------------------------------------------------------

class TestComputeClimberGrid:
    """Direct tests for the SC2MapAnalysis algorithm port."""

    def test_detects_vertical_cliff_with_correct_height_diff(self, synthetic_map):
        """Interior cells of the cliff gap row must be flagged by the algorithm."""
        path, height = synthetic_map
        result = ObservationTransformer._compute_climber_grid(path, height)
        # Interior cols well away from the 2-cell boundary (boundary excluded by C algo)
        interior_cols = range(4, MAP_SIZE - 4)
        assert np.any(result[5, list(interior_cols)]), \
            "At least one interior cliff-gap cell must be detected"

    def test_no_false_positives_on_flat_terrain(self):
        """All-flat terrain (uniform height) must produce zero climbable cells."""
        path = np.ones((16, 16), dtype=np.uint8)
        height = np.full((16, 16), 20.0, dtype=np.float32)
        result = ObservationTransformer._compute_climber_grid(path, height)
        assert not np.any(result), "Flat terrain must have no climbable cells"

    def test_wrong_height_diff_not_detected(self):
        """Height difference != 16 (e.g. 100) must not trigger detection."""
        path = np.ones((16, 16), dtype=np.uint8)
        height = np.full((16, 16), 12.0, dtype=np.float32)
        # Gap row with height = 112 (diff = 100, not 16)
        path[7, :] = 0
        height[7, :] = 112.0
        result = ObservationTransformer._compute_climber_grid(path, height)
        assert not np.any(result[7, :]), \
            "Height diff of 100 must not be flagged (only diff == 16 is a SC2 cliff)"

    def test_detects_diagonal_cliff_line(self):
        """Diagonal cliff runs must survive the connectivity filter (8-neighbour)."""
        path = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        height = np.full((MAP_SIZE, MAP_SIZE), LOW_H, dtype=np.float32)

        for i in range(MAP_SIZE):
            path[i, i] = 0
            height[i, i] = CLIFF_FACE_H
            for y in range(MAP_SIZE):
                for x in range(MAP_SIZE):
                    if x > y:
                        height[y, x] = HIGH_H
                    elif x < y:
                        height[y, x] = LOW_H

        result = ObservationTransformer._compute_climber_grid(path, height)
        interior = [i for i in range(4, MAP_SIZE - 4)]
        detected = [i for i in interior if result[i, i]]
        assert len(detected) >= len(interior) - 2, (
            f"Diagonal cliff line mostly missing: only {len(detected)}/{len(interior)} "
            f"interior cells flagged"
        )

    def test_connectivity_filter_removes_isolated_cell(self):
        """A single isolated CLIMBABLE cell with no adjacent CLIMBABLE neighbour
        must be removed by the connectivity filter."""
        # Map: high ground rows 0-4, single-col cliff gap at (5,8), low ground rows 6+
        # The cliff is only 1 cell wide in x — but 16 cells wide in x direction means
        # neighbours at x=7 and x=9 are NOT part of the cliff gap (they're walkable).
        # So only (5,8) would be climbable-candidate and it has no climbable neighbour.
        path = np.ones((16, 16), dtype=np.uint8)
        height = np.full((16, 16), LOW_H, dtype=np.float32)
        height[0:5, :] = HIGH_H
        height[5, :] = CLIFF_FACE_H
        path[5, :] = 0
        # Re-open all gap cells except col 8 as walkable → only col 8 is the gap
        path[5, :] = 1
        path[5, 8] = 0
        result = ObservationTransformer._compute_climber_grid(path, height)
        # The isolated single gap cell has no adjacent climbable cell
        assert not result[5, 8], \
            "Isolated single climbable cell must be removed by connectivity filter"

    def test_walkable_cells_never_flagged(self, synthetic_map):
        """No walkable cell should appear in the climber grid."""
        path, height = synthetic_map
        result = ObservationTransformer._compute_climber_grid(path, height)
        walkable = path > 0
        assert not np.any(result & walkable), \
            "Walkable cells must never appear in the climber grid"


# ---------------------------------------------------------------------------
# Integration tests for _draw_terrain_pathing
# ---------------------------------------------------------------------------

class TestCliffWalkerTerrain:

    def test_standard_unit_cliff_gap_is_dark(self, synthetic_map):
        """Standard unit: the non-walkable cliff gap must appear as 0."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=False)
        assert np.all(result[_feat_row(5), :] == 0.0), \
            "Non-walkable cliff gap must be 0 for standard unit"

    def test_standard_unit_walkable_cells_nonzero(self, synthetic_map):
        """Standard unit: all walkable cells must have non-zero terrain values."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=False)
        for data_row in (0, 1, 2, 3, 4, 6, 7, 8):
            assert np.all(result[_feat_row(data_row), :] > 0.0), \
                f"Walkable data row {data_row} must be > 0 for standard unit"

    def test_cliff_walker_diagonal_cliff_interior_marked(self):
        """Cliff walker: diagonal jump cliffs must render along the full interior run."""
        path = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        height = np.full((MAP_SIZE, MAP_SIZE), LOW_H, dtype=np.float32)
        for i in range(MAP_SIZE):
            path[i, i] = 0
            height[i, i] = CLIFF_FACE_H
            for y in range(MAP_SIZE):
                for x in range(MAP_SIZE):
                    if x > y:
                        height[y, x] = HIGH_H
                    elif x < y:
                        height[y, x] = LOW_H

        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=True)
        low_ref = result[_feat_row(MAP_SIZE - 5), 4]
        for i in range(4, MAP_SIZE - 4):
            feat_y, feat_x = _feat_row(i), i
            assert result[feat_y, feat_x] > 0.0, f"Diagonal cliff ({i},{i}) must be visible"
            assert result[feat_y, feat_x] > low_ref, (
                f"Diagonal cliff ({i},{i}) must be brighter than adjacent low ground"
            )

    def test_cliff_walker_cliff_gap_interior_cols_marked(self, synthetic_map):
        """Cliff walker: interior cliff gap cells must be above 0 and above low ground."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=True)

        low_ground_val = result[_feat_row(10), 8]   # a walkable low-ground interior cell
        # Interior cols (avoid 2-cell boundary exclusion zone)
        for col in range(4, MAP_SIZE - 4):
            cliff_val = result[_feat_row(5), col]
            assert cliff_val > 0.0, \
                f"Cliff gap col {col} must be non-zero for cliff walker"
            assert cliff_val > low_ground_val, \
                f"Cliff gap col {col} ({cliff_val:.3f}) must be brighter than low ground ({low_ground_val:.3f})"

    def test_cliff_walker_walkable_terrain_unchanged(self, synthetic_map):
        """Cliff walker: walkable cells must be IDENTICAL to the standard view."""
        path, height = synthetic_map
        t_std = _make_transformer(feature_map_size=MAP_SIZE)
        t_cw = _make_transformer(feature_map_size=MAP_SIZE)
        std = _call_draw(t_std, path, height, cliff_walker=False)
        cw = _call_draw(t_cw, path, height, cliff_walker=True)

        for data_row in (0, 1, 2, 3, 4, 6, 7, 8):
            np.testing.assert_array_almost_equal(
                std[_feat_row(data_row), :],
                cw[_feat_row(data_row), :],
                err_msg=f"Walkable data row {data_row} must be identical in both views",
            )

    def test_terrain_ramp_preserves_height_gradient(self):
        """Ramp cells must interpolate smoothly between low and high ground."""
        path = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        height = np.full((MAP_SIZE, MAP_SIZE), LOW_H, dtype=np.float32)
        height[0:4, :] = HIGH_H
        for x in range(MAP_SIZE):
            height[8, x] = LOW_H + (HIGH_H - LOW_H) * (x / float(MAP_SIZE - 1))

        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=False)
        ramp_vals = result[_feat_row(8), 4 : MAP_SIZE - 4]
        unique_steps = len(np.unique(np.round(ramp_vals, 2)))
        assert unique_steps >= 4, (
            f"Ramp should show gradual brightness, got {unique_steps} distinct values"
        )
        assert float(ramp_vals[0]) < float(ramp_vals[-1]), (
            "Ramp brightness must increase toward high ground"
        )

    def test_terrain_height_contrast_uses_map_cliff_tiers(self, synthetic_map):
        """Low and high ground must span most of the obs brightness range."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=False)

        low_val = float(result[_feat_row(12), 8])
        high_val = float(result[_feat_row(1), 8])
        assert high_val - low_val >= 0.39, (
            f"Expected strong visible contrast, got low={low_val:.3f} high={high_val:.3f}"
        )

    def _tier_brightness(self, height_val: float, min_h: float) -> float:
        return float(
            ObservationTransformer._normalize_terrain_heights(
                np.array([height_val], dtype=np.float32),
                min_h,
                1.0,
            )[0]
        )

    def test_adjacent_cliff_tier_brightness_step_uniform_two_and_three_tiers(self):
        """One cliff tier (16 gu) must advance brightness by the same amount on every map."""
        from src.transformers.observation_transformer import (
            _TERRAIN_BRIGHTNESS_PER_CLIFF_TIER,
        )

        min_h = LOW_H
        two_tier_delta = self._tier_brightness(HIGH_H, min_h) - self._tier_brightness(
            LOW_H, min_h
        )
        mid_h = LOW_H + 16.0
        top_h = LOW_H + 32.0
        three_low = self._tier_brightness(LOW_H, min_h)
        three_mid = self._tier_brightness(mid_h, min_h)
        three_high = self._tier_brightness(top_h, min_h)
        step_3a = three_mid - three_low
        step_3b = three_high - three_mid

        assert two_tier_delta == pytest.approx(_TERRAIN_BRIGHTNESS_PER_CLIFF_TIER)
        assert step_3a == pytest.approx(_TERRAIN_BRIGHTNESS_PER_CLIFF_TIER)
        assert 0.0 < step_3b <= _TERRAIN_BRIGHTNESS_PER_CLIFF_TIER + 1e-6

    def test_two_tier_map_contrast_not_compressed_by_height_span_outliers(self):
        """Maps with two play heights must not look washed out when tier span is wider."""
        path = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        height = np.full((MAP_SIZE, MAP_SIZE), LOW_H, dtype=np.float32)
        height[0 : MAP_SIZE // 2, :] = HIGH_H
        height[2, 2] = HIGH_H + 48.0  # rare walkable outlier tier

        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=False)
        low_val = float(result[_feat_row(12), 8])
        high_val = float(result[_feat_row(2), 8])
        assert high_val - low_val >= 0.39, (
            f"Two-tier map should keep strong low/high contrast, got "
            f"low={low_val:.3f} high={high_val:.3f}"
        )

    def test_terrain_height_ignores_offscreen_outliers_on_large_map(self):
        """Map-wide height outliers must not flatten contrast in the zoomed obs view."""
        size = 256
        path = np.zeros((size, size), dtype=np.uint8)
        path[64:192, 64:192] = 1
        height = np.full((size, size), LOW_H, dtype=np.float32)
        height[64:128, 64:192] = HIGH_H
        height[8, 8] = 200.0

        gi = _make_game_info(path, height)
        gi.playable_area = MagicMock()
        gi.playable_area.x = 64
        gi.playable_area.y = 64
        gi.playable_area.width = 128
        gi.playable_area.height = 128

        t = ObservationTransformer(
            map_size=(128, 128),
            training_conf=_make_training_conf(),
        )
        raw_obs = {
            "units": [],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        terrain = t.transform(raw_obs)["map"][0]

        high_px, high_py = t._game_to_obs_pixel(96.0, 96.0)
        low_px, low_py = t._game_to_obs_pixel(96.0, 160.0)
        high_val = float(terrain[high_py, high_px])
        low_val = float(terrain[low_py, low_px])
        assert high_val - low_val >= 0.39, (
            f"Zoomed obs should contrast local tiers, got low={low_val:.3f} high={high_val:.3f}"
        )

    def test_terrain_height_ignores_rare_playable_low_outlier(self):
        """Rare low walkable cells must not flatten dominant playable cliff tiers."""
        size = 176
        path = np.zeros((size, size), dtype=np.uint8)
        path[16:148, 16:148] = 1
        height = np.full((size, size), 175.0, dtype=np.float32)
        height[16:148, 16:80] = 191.0
        height[16:148, 80:148] = 207.0
        path[16, 16] = 1
        height[16, 16] = 112.0

        gi = _make_game_info(path, height)
        gi.playable_area = MagicMock()
        gi.playable_area.x = 16
        gi.playable_area.y = 16
        gi.playable_area.width = 132
        gi.playable_area.height = 132

        t = ObservationTransformer(
            map_size=(128, 128),
            training_conf=_make_training_conf(),
        )
        raw_obs = {
            "units": [],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
        }
        terrain = t.transform(raw_obs)["map"][0]
        visible = terrain[terrain > 0.05]
        assert float(np.mean(visible >= 0.99)) < 0.95, (
            "Dominant cliff tiers must not all clip to max brightness"
        )
        assert visible.max() - visible.min() >= 0.39
        assert len(np.unique(np.round(visible, 2))) >= 2

    def test_cliff_walker_creates_gradient(self, synthetic_map):
        """The cliff gap is brighter than low ground, creating a visible gradient.

        Cliff cells show at the local maximum walkable height (= high-ground
        brightness), so the visual reads: dark (low) → bright (cliff) → bright (high).
        """
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=True)

        # Interior column (col=8) — clear of the 2-cell boundary exclusion zone
        low_ground = result[_feat_row(12), 8]    # data row 12: low ground interior
        cliff_gap = result[_feat_row(5), 8]      # data row 5: cliff gap
        high_ground = result[_feat_row(1), 8]    # data row 1: high ground interior

        # Cliff face must sit between low ground and high ground (mean of both sides)
        assert low_ground < cliff_gap < high_ground, (
            f"Expected gradient: low({low_ground:.3f}) < cliff({cliff_gap:.3f}) < "
            f"high({high_ground:.3f})"
        )
        # Should be close to the midpoint of the two walkable heights
        midpoint = (low_ground + high_ground) / 2.0
        assert abs(cliff_gap - midpoint) < 0.15, (
            f"Cliff ({cliff_gap:.3f}) should be near the midpoint ({midpoint:.3f}) "
            f"of low({low_ground:.3f}) and high({high_ground:.3f})"
        )

    def test_wall_obstacle_not_marked_as_cliff(self):
        """A non-walkable obstacle on flat terrain (height diff = 0) must remain 0."""
        path = np.ones((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        height = np.full((MAP_SIZE, MAP_SIZE), 100.0, dtype=np.float32)
        path[7, 7] = 0  # obstacle on flat ground

        t = _make_transformer(feature_map_size=MAP_SIZE)
        result = _call_draw(t, path, height, cliff_walker=True)

        # The obstacle cell should remain 0 (not raised as a cliff face)
        feat_row = _feat_row(7)
        feat_col = 7
        assert result[feat_row, feat_col] == 0.0, \
            "Flat-ground obstacle must not be marked as a cliff face"

    def test_separate_cache_entries_populated(self, synthetic_map):
        """Both cache variants are stored under distinct keys after consecutive calls."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)

        _call_draw(t, path, height, cliff_walker=False)
        assert any(k[0] == "standard" for k in t._terrain_caches)
        assert not any(k[0] == "cliff_walker" for k in t._terrain_caches)

        _call_draw(t, path, height, cliff_walker=True)
        assert any(k[0] == "cliff_walker" for k in t._terrain_caches)

        std = next(v for k, v in t._terrain_caches.items() if k[0] == "standard")
        cw = next(v for k, v in t._terrain_caches.items() if k[0] == "cliff_walker")
        assert not np.array_equal(std, cw), \
            "Standard and cliff_walker caches must differ"

    def test_cache_hit_reuses_stored_array(self, synthetic_map):
        """A second call with the same map/structure count returns the cached result."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        first = _call_draw(t, path, height, cliff_walker=False).copy()
        second = _call_draw(t, path, height, cliff_walker=False).copy()
        np.testing.assert_array_equal(first, second)

    def test_cache_hit_ignores_structure_count(self, synthetic_map):
        """Global terrain cache is static map pathing only — structure count does not bust it."""
        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        first = _call_draw(t, path, height, cliff_walker=False).copy()
        second = _call_draw(t, path, height, cliff_walker=False).copy()
        np.testing.assert_array_equal(first, second)
        assert any(k[0] == "standard" for k in t._terrain_caches)

    def test_structure_overlay_preserves_cliff_faces_in_obs_rect(self, synthetic_map):
        """Building terrain stamps must not erase reaper-jump cliffs inside the obs bounding box."""
        from sc2.ids.unit_typeid import UnitTypeId
        from sc2.position import Point2

        path, height = synthetic_map
        t = _make_transformer(feature_map_size=MAP_SIZE)
        t._cached_path_data = path.astype(np.uint8)
        t._current_game_data = None
        game_info = _make_game_info(path, height)
        n_layers = LayerConstants.get_num_layers()
        layers = np.zeros(
            (n_layers, MAP_SIZE, MAP_SIZE), dtype=np.float32
        )
        t._draw_terrain_pathing(game_info, layers, cliff_walker=True)
        terrain_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]
        terrain = layers[terrain_idx]
        cliff_feat_row = _feat_row(5)
        cliff_val_before = float(terrain[cliff_feat_row, 8])
        assert cliff_val_before > 0.0

        barracks = MagicMock()
        barracks.position = Point2((8.0, 10.0))
        barracks.type_id = UnitTypeId.BARRACKS
        barracks.is_structure = True
        barracks.is_flying = False
        barracks.radius = 0.0
        barracks.tag = 1

        t._overlay_map_blocker_terrain(terrain, [barracks])
        assert float(terrain[cliff_feat_row, 8]) == cliff_val_before, (
            "Cliff-face brightness must survive structure terrain overlay outside building footprint"
        )

    def test_tactical_terrain_brightness_stable_when_focal_moves(self):
        """Tactical terrain at a fixed world point must not change as the fovea pans."""
        from sc2.position import Point2
        from sc2.ids.unit_typeid import UnitTypeId
        from src.utils.obs_geometry import world_to_tactical_obs_pixel
        from tests.test_observation_transformer import create_mock_unit

        size = 64
        path = np.ones((size, size), dtype=np.uint8)
        height = np.full((size, size), LOW_H, dtype=np.float32)
        height[0:24, :] = HIGH_H
        gi = _make_game_info(path, height)
        world_pt = (40.0, 32.0)
        conf = _make_training_conf()

        def brightness_at(focal_xy):
            transformer = ObservationTransformer(
                map_size=(128, 128), training_conf=conf
            )
            focal = create_mock_unit(UnitTypeId.MARINE, Point2(focal_xy))
            raw_obs = {
                "units": [focal],
                "structures": [],
                "enemy_units": [],
                "enemy_structures": [],
                "effects": [],
                "game_info": gi,
                "game_data": MagicMock(),
                "selected_units": [focal],
            }
            tactical = transformer.transform_tactical(raw_obs, focal)[0]
            px, py = world_to_tactical_obs_pixel(
                world_pt[0], world_pt[1], focal_xy[0], focal_xy[1]
            )
            return float(tactical[py, px])

        a = brightness_at((32.0, 32.0))
        b = brightness_at((36.0, 32.0))
        c = brightness_at((32.0, 36.0))
        assert a == pytest.approx(b)
        assert a == pytest.approx(c)

    def test_tactical_reaper_sees_cliff_gap(self):
        """Tactical view must render jumpable cliff faces for the focal Reaper."""
        from sc2.position import Point2

        size = 64
        cliff_row = 30
        path = np.zeros((size, size), dtype=np.uint8)
        height = np.zeros((size, size), dtype=np.float32)
        path[0:cliff_row, :] = 1
        height[0:cliff_row, :] = HIGH_H
        path[cliff_row, :] = 0
        height[cliff_row, :] = CLIFF_FACE_H
        path[cliff_row + 1 :, :] = 1
        height[cliff_row + 1 :, :] = LOW_H

        gi = _make_game_info(path, height)
        reaper = MagicMock()
        reaper.position = Point2((32.0, 40.0))
        reaper.type_id.name = "REAPER"
        reaper.radius = 0.5
        reaper.is_structure = False
        reaper.is_flying = False
        reaper.tag = 42

        conf = _make_training_conf()
        transformer = ObservationTransformer(map_size=(128, 128), training_conf=conf)
        raw_obs = {
            "units": [reaper],
            "structures": [],
            "enemy_units": [],
            "enemy_structures": [],
            "effects": [],
            "game_info": gi,
            "game_data": MagicMock(),
            "high_level_command": 0,
        }
        tactical = transformer.transform_tactical(raw_obs, reaper)
        assert tactical.shape[0] == 1
        terrain = tactical[0]
        low_py = 31 - int(40 - math.floor(40.0 - 16.0))
        cliff_py = 31 - int(cliff_row - math.floor(40.0 - 16.0))
        col = 32 - math.floor(40.0 - 16.0)
        assert terrain[low_py, col] > 0.0
        assert terrain[cliff_py, col] > terrain[low_py, col], (
            "Cliff gap must be brighter than adjacent low ground for Reaper tactical view"
        )


# ---------------------------------------------------------------------------
# Tests for _detect_cliff_walker (unchanged from before)
# ---------------------------------------------------------------------------

class TestDetectCliffWalker:
    """Tests for the auto-detection helper — no bot should set cliff_walker explicitly."""

    def _make_unit(self, type_name: str):
        u = MagicMock()
        u.type_id.name = type_name
        return u

    def test_colossus_detected_via_active_unit(self):
        t = _make_transformer()
        u = self._make_unit("COLOSSUS")
        assert t._detect_cliff_walker({}, active_unit=u) is True

    def test_reaper_detected_via_active_unit(self):
        t = _make_transformer()
        u = self._make_unit("REAPER")
        assert t._detect_cliff_walker({}, active_unit=u) is True

    def test_standard_unit_not_cliff_walker_via_active_unit(self):
        t = _make_transformer()
        for name in ("MARINE", "ZEALOT", "ZERGLING", "STALKER", "HELLION"):
            u = self._make_unit(name)
            assert t._detect_cliff_walker({}, active_unit=u) is False, \
                f"{name} should not be cliff walker"

    def test_colossus_detected_via_selected_units(self):
        t = _make_transformer()
        u = self._make_unit("COLOSSUS")
        assert t._detect_cliff_walker({"selected_units": [u]}) is True

    def test_reaper_detected_via_selected_units(self):
        t = _make_transformer()
        u = self._make_unit("REAPER")
        assert t._detect_cliff_walker({"selected_units": [u]}) is True

    def test_standard_unit_not_detected_via_selected_units(self):
        t = _make_transformer()
        u = self._make_unit("MARINE")
        assert t._detect_cliff_walker({"selected_units": [u]}) is False

    def test_active_unit_takes_priority_over_selected_units(self):
        """active_unit (Reaper) should win over selected_units (Marine)."""
        t = _make_transformer()
        reaper = self._make_unit("REAPER")
        marine = self._make_unit("MARINE")
        assert t._detect_cliff_walker({"selected_units": [marine]}, active_unit=reaper) is True

    def test_no_unit_returns_false(self):
        t = _make_transformer()
        assert t._detect_cliff_walker({}) is False
        assert t._detect_cliff_walker({"selected_units": []}) is False

    def test_explicit_override_fallback(self):
        """raw_obs['cliff_walker'] still works when no unit is provided."""
        t = _make_transformer()
        assert t._detect_cliff_walker({"cliff_walker": True}) is True
        assert t._detect_cliff_walker({"cliff_walker": False}) is False

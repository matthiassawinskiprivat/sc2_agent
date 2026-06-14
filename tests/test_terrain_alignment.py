import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


class MockGrid:
    def __init__(self, data):
        self.data_numpy = data


class TestTerrainAlignment(unittest.TestCase):
    def setUp(self):
        self.f_w, self.f_h = 128, 128
        self.transformer = ObservationTransformer(map_size=(self.f_w, self.f_h))

    def create_mock_game_info(self, width, height):
        info = MagicMock()
        info.map_size.width = width
        info.map_size.height = height
        info.map_name = "TestMap"

        info.playable_area = MagicMock()
        info.playable_area.x = 0
        info.playable_area.y = 0
        info.playable_area.width = width
        info.playable_area.height = height

        # SC2 pathing grids are (map_height, map_width).
        shape = (height, width)
        self.pathing = np.zeros(shape, dtype=np.uint8)
        self.height = np.zeros(shape, dtype=np.float32)

        info.pathing_grid = MockGrid(self.pathing)
        info.terrain_height = MockGrid(self.height)
        return info

    def _terrain_layer(self, raw_obs):
        result = self.transformer.transform(raw_obs)
        layers = result["map"] if isinstance(result, dict) else result
        return layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]]

    def test_rectangular_map_letterboxing_wide(self):
        map_w, map_h = 200, 100
        game_info = self.create_mock_game_info(map_w, map_h)
        self.pathing[:, :] = 1
        self.height[:, :] = 10.0

        raw_obs = {
            "game_info": game_info,
            "creep": MockGrid(np.zeros_like(self.pathing)),
        }

        terrain_layer = self._terrain_layer(raw_obs)

        expected_scale = 128 / 200
        self.assertAlmostEqual(self.transformer.scale, expected_scale)
        self.assertEqual(self.transformer.offset_y, 32)
        self.assertEqual(self.transformer.offset_x, 0)

        self.assertEqual(np.sum(terrain_layer[0:30, :]), 0)
        self.assertEqual(np.sum(terrain_layer[-30:, :]), 0)
        self.assertGreater(np.sum(terrain_layer[64, :]), 0)

    def test_creep_exact_alignment(self):
        map_size = 100
        game_info = self.create_mock_game_info(map_size, map_size)
        self.pathing[:, :] = 1
        self.height[:, :] = 10.0
        self.pathing[50, 50] = 1

        creep_grid = np.zeros((map_size, map_size), dtype=np.uint8)
        creep_grid[50, 50] = 1

        raw_obs = {
            "game_info": game_info,
            "creep": MockGrid(creep_grid),
        }

        result = self.transformer.transform(raw_obs)
        layers = result["map"] if isinstance(result, dict) else result
        path_layer = layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]]
        creep_layer = layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_CREEP]]

        self.assertGreater(np.sum(path_layer), 0.0)
        self.assertGreater(np.sum(creep_layer), 0.0)
        self.assertGreater(float(creep_layer.max()), 0.0)

    def test_corner_pixel_visible_on_rectangular_map(self):
        w, h = 100, 50
        game_info = self.create_mock_game_info(w, h)
        self.pathing[:, :] = 1
        self.height[:, :] = 10.0
        self.pathing[49, 99] = 1

        raw_obs = {"game_info": game_info}
        terrain_layer = self._terrain_layer(raw_obs)

        found = False
        for y in range(31, 35):
            for x in range(124, 128):
                if terrain_layer[y, x] > 0:
                    found = True

        self.assertTrue(
            found,
            f"Top-right corner pixel missing. Scale: {self.transformer.scale}, "
            f"Offsets: {self.transformer.offset_x},{self.transformer.offset_y}",
        )


if __name__ == "__main__":
    unittest.main()

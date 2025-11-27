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

    def create_mock_game_info(self, width, height, data_shape_transposed=False):
        info = MagicMock()
        info.map_size.width = width
        info.map_size.height = height
        info.map_name = "TestMap"

        # FIX: Define playable_area matching map size for these tests
        info.playable_area.x = 0
        info.playable_area.y = 0
        info.playable_area.width = width
        info.playable_area.height = height

        shape = (height, width) if data_shape_transposed else (width, height)
        self.pathing = np.zeros(shape, dtype=np.uint8)
        self.height = np.zeros(shape, dtype=np.float32)

        info.pathing_grid = MockGrid(self.pathing)
        info.terrain_height = MockGrid(self.height)
        return info

    def test_rectangular_map_letterboxing_wide(self):
        map_w, map_h = 200, 100
        game_info = self.create_mock_game_info(map_w, map_h)
        self.pathing[:, :] = 1
        self.height[:, :] = 10.0

        raw_obs = {
            'game_info': game_info,
            'creep': MockGrid(np.zeros_like(self.pathing))
        }

        layers = self.transformer.transform(raw_obs)
        terrain_layer = layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]]

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
        self.pathing[50, 50] = 1

        creep_grid = np.zeros((map_size, map_size), dtype=np.uint8)
        creep_grid[50, 50] = 1

        raw_obs = {
            'game_info': game_info,
            'creep': MockGrid(creep_grid)
        }

        layers = self.transformer.transform(raw_obs)
        path_layer = layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]]
        creep_layer = layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_CREEP]]

        py_path, px_path = np.where(path_layer > 0)
        py_creep, px_creep = np.where(creep_layer > 0)

        self.assertTrue(len(py_path) > 0)
        self.assertTrue(len(py_creep) > 0)
        self.assertEqual(list(py_path), list(py_creep))
        self.assertEqual(list(px_path), list(px_creep))

    def test_transposed_grid_handling(self):
        w, h = 100, 50
        game_info = self.create_mock_game_info(w, h, data_shape_transposed=True)
        self.pathing[49, 99] = 1
        self.height[49, 99] = 10

        raw_obs = {'game_info': game_info}
        layers = self.transformer.transform(raw_obs)

        terrain_layer = layers[LayerConstants.LAYERS[LayerConstants.TERRAIN_PATHABLE]]

        found = False
        for y in range(38, 41):
            for x in range(112, 115):
                if terrain_layer[y, x] > 0:
                    found = True

        self.assertTrue(found)


if __name__ == '__main__':
    unittest.main()
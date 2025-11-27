import unittest
import numpy as np
import sys
import os
import json
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from sc2.ids.unit_typeid import UnitTypeId


class MockPoint2:
    def __init__(self, x, y): self.x, self.y = x, y


class MockUnit:
    def __init__(self, type_id, x, y):
        self.type_id = type_id
        self.position = MockPoint2(x, y)
        self.radius = 1
        self.is_structure = False
        self.is_cloaked = False
        self.is_detected = True
        self.is_snapshot = False
        self.name = "TestUnit"
        # Minimal attributes
        self.is_blip = False
        self.health_max = 100
        self.shield_max = 0
        self.health = 100
        self.shield = 0
        self.energy = 0
        self.energy_max = 0
        self.is_biological = False
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.is_flying = False
        self.detect_range = 0
        self.is_ready = True
        self.is_hallucination = False
        self.weapon_cooldown = 0


class TestRobustness(unittest.TestCase):

    def test_json_load_failure(self):
        """If unit_data.json is missing/corrupt, init should not crash."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            transformer = ObservationTransformer(map_size=(64, 64))
            self.assertEqual(transformer.fallback_db, {})

    def test_json_load_success(self):
        """Verify fallback DB is loaded correctly."""
        dummy_data = json.dumps({"MARINE": {"weapons": []}})
        with patch("builtins.open", mock_open(read_data=dummy_data)):
            transformer = ObservationTransformer(map_size=(64, 64))
            self.assertIn("MARINE", transformer.fallback_db)

    def test_out_of_bounds_unit(self):
        """Units outside the feature map view should be ignored."""
        transformer = ObservationTransformer(map_size=(100, 100))
        game_info = MagicMock()
        game_info.map_size.width = 100
        game_info.map_size.height = 100

        # FIX: Explicitly create playable_area mock
        playable_area = MagicMock()
        playable_area.x = 0
        playable_area.y = 0
        playable_area.width = 100
        playable_area.height = 100
        game_info.playable_area = playable_area

        # Create minimal grids
        game_info.pathing_grid.data_numpy = np.zeros((100, 100))
        game_info.terrain_height.data_numpy = np.zeros((100, 100))

        # Unit at (200, 200) -> Way outside 100x100
        u = MockUnit(UnitTypeId.MARINE, 200, 200)

        raw_obs = {
            'units': [u],
            'enemy_units': [],
            'game_info': game_info
        }

        layers = transformer.transform(raw_obs)
        self.assertEqual(np.sum(layers), 0)

    def test_unit_without_position(self):
        """Unit with None position (e.g. inside transport) should be skipped."""
        u = MockUnit(UnitTypeId.MARINE, 0, 0)
        u.position = None

        transformer = ObservationTransformer()
        game_info = MagicMock()
        game_info.map_size.width = 128
        game_info.map_size.height = 128

        # FIX: Explicitly create playable_area mock
        playable_area = MagicMock()
        playable_area.x = 0
        playable_area.y = 0
        playable_area.width = 128
        playable_area.height = 128
        game_info.playable_area = playable_area

        game_info.pathing_grid.data_numpy = np.zeros((128, 128))
        game_info.terrain_height.data_numpy = np.zeros((128, 128))

        layers = transformer.transform({'units': [u], 'game_info': game_info})
        self.assertEqual(np.sum(layers), 0)


if __name__ == '__main__':
    unittest.main()
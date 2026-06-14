import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from tests.obs_transformer_test_utils import get_layer, transform_map


class MockPoint2:
    def __init__(self, x, y): self.x, self.y = x, y


class MockTypeId:
    def __init__(self, name): self.name = name

    @property
    def value(self): return 999


class MockUnit:
    def __init__(self, name, x, y, radius=1.0, is_structure=False):
        self.type_id = MockTypeId(name)
        self.position = MockPoint2(x, y)
        self.radius = radius
        self.is_structure = is_structure

        # Defaults
        self.alliance = 1
        self.is_cloaked = False
        self.is_detected = True
        self.is_blip = False
        self.is_flying = False
        self.is_biological = False
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.health_max = 100
        self.shield_max = 0
        self.health = 100
        self.shield = 0
        self.energy_max = 0
        self.energy = 0
        self.weapon_cooldown = 0
        self.detect_range = 0
        self.build_progress = 1.0
        self.is_hallucination = False


class TestStructuresAndNeutrals(unittest.TestCase):
    def setUp(self):
        self.transformer = ObservationTransformer(map_size=(128, 128))
        self.game_info = MagicMock()
        self.game_info.map_size.width = 128
        self.game_info.map_size.height = 128

        # FIX: Explicitly create playable_area with integer values
        # This prevents MagicMock vs int comparison errors in the transformer
        playable_area = MagicMock()
        playable_area.x = 0
        playable_area.y = 0
        playable_area.width = 128
        playable_area.height = 128
        self.game_info.playable_area = playable_area

        self.game_info.pathing_grid = MagicMock()
        self.game_info.pathing_grid.data_numpy = np.zeros((128, 128))
        self.game_info.terrain_height = MagicMock()
        self.game_info.terrain_height.data_numpy = np.zeros((128, 128))

    def test_exclude_minerals(self):
        """Ensure units with 'MINERAL' in name are ignored."""
        mineral = MockUnit("LabMineralField", 64, 64)
        marine = MockUnit("Marine", 60, 60)

        raw_obs = {
            'units': [marine],
            'enemy_units': [mineral],  # Should be ignored
            'game_info': self.game_info
        }

        map_data = transform_map(self.transformer, raw_obs)

        self.assertGreater(
            np.sum(get_layer(map_data, LayerConstants.ALLIED_UNIT_DENSITY)), 0,
            "Marine should appear",
        )
        self.assertEqual(
            np.sum(get_layer(map_data, LayerConstants.ENEMY_UNIT_DENSITY)), 0,
            "Mineral should NOT appear as enemy",
        )

    def test_structure_radius_scaling(self):
        """
        Compare a Marine (Radius ~0.5) vs Command Center (Radius 2.75).
        CC should occupy significantly more pixels.
        """
        # Marine at (40, 40)
        marine = MockUnit("Marine", 40, 40, radius=0.375, is_structure=False)

        # CC at (80, 80)
        cc = MockUnit("CommandCenter", 80, 80, radius=2.75, is_structure=True)

        raw_obs = {
            'units': [marine, cc],
            'game_info': self.game_info
        }

        map_data = transform_map(self.transformer, raw_obs)
        density_layer = get_layer(map_data, LayerConstants.ALLIED_UNIT_DENSITY)

        # Count non-zero pixels around Marine
        # (Naive slice around 40,40)
        marine_pixels = np.count_nonzero(density_layer[80:100, 30:50])  # Image coords are flipped Y

        # Count non-zero pixels around CC
        cc_pixels = np.count_nonzero(density_layer[40:60, 70:90])

        # print(f"Marine Pixels: {marine_pixels}, CC Pixels: {cc_pixels}")

        self.assertGreater(cc_pixels, marine_pixels * 3, "Command Center should be much larger than Marine")

    def test_integer_type_id_compatibility(self):
        """
        Ensure passing an integer as type_id (common in some tests/mocks)
        doesn't crash the 'unit.type_id.name' check.
        """
        u = MockUnit("SomeUnit", 50, 50)
        u.type_id = 135  # ForceField ID (int)

        raw_obs = {'units': [u], 'game_info': self.game_info}

        try:
            self.transformer.transform(raw_obs)
        except AttributeError:
            self.fail("Transformer crashed on integer type_id")


if __name__ == '__main__':
    unittest.main()
import unittest
import sys
import os
import json
import numpy as np
from unittest.mock import MagicMock
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.effect_id import EffectId

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


class MockGrid:
    def __init__(self, width, height, fill_value=0, dtype=np.uint8):
        self.width = width
        self.height = height
        self.data_numpy = np.full((width, height), fill_value, dtype=dtype)


class MockPoint2:
    def __init__(self, x, y): self.x, self.y = x, y

    def distance_to_point2(self, p): return ((self.x - p.x) ** 2 + (self.y - p.y) ** 2) ** 0.5

    def __getitem__(self, item): return self.x if item == 0 else self.y


class MockUnitType:
    def __init__(self, name, value=0):
        self.name = name
        self.value = value


class MockUnit:
    def __init__(self, name, x=100, y=100, alliance=1,
                 is_light=False, is_armored=False, is_biological=False, is_mechanical=False,
                 is_massive=False, is_psionic=False,
                 is_flying=False, is_cloaked=False, is_detected=True,
                 energy=0, cooldown=0, detect_range=0, build_progress=1.0):

        try:
            self.type_id = UnitTypeId[name]
        except (KeyError, AttributeError):
            self.type_id = MockUnitType(name)

        self.position = MockPoint2(x, y)
        self.alliance = alliance
        self.radius = 0.5

        self.is_structure = False
        self.is_light = is_light
        self.is_armored = is_armored
        self.is_biological = is_biological
        self.is_mechanical = is_mechanical
        self.is_massive = is_massive
        self.is_psionic = is_psionic
        self.is_flying = is_flying
        self.is_cloaked = is_cloaked
        self.is_detected = is_detected
        self.is_blip = False
        self.is_snapshot = False
        self.health_max = 100
        self.shield_max = 50
        self.health = 100
        self.shield = 50
        self.energy = energy
        self.energy_max = 200 if energy > 0 else 0
        self.weapon_cooldown = cooldown
        self.detect_range = detect_range
        self.attack_upgrade_level = 0
        self.build_progress = build_progress
        self.is_ready = (build_progress == 1.0)
        self.is_hallucination = False


class MockEffect:
    def __init__(self, id_val, x, y):
        self.id = id_val
        self.positions = [MockPoint2(x, y)]


class TestUnitLayerValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'unit_data.json')
        if not os.path.exists(cls.data_path):
            cls.data_path = os.path.join(os.getcwd(), 'data', 'unit_data.json')

    def setUp(self):
        self.transformer = ObservationTransformer(map_size=(128, 128), unit_data_file=self.data_path)
        self.map_w, self.map_h = 200, 200

        self.game_info = MagicMock()
        self.game_info.map_size.width = self.map_w
        self.game_info.map_size.height = self.map_h

        # Explicit playable area
        playable_area = MagicMock()
        playable_area.x = 0
        playable_area.y = 0
        playable_area.width = self.map_w
        playable_area.height = self.map_h
        self.game_info.playable_area = playable_area

        self.game_info.pathing_grid = MockGrid(self.map_w, self.map_h, fill_value=1)
        height_data = np.fromfunction(lambda i, j: (i + j) % 10, (self.map_w, self.map_h), dtype=np.float32)
        self.game_info.terrain_height = MockGrid(self.map_w, self.map_h, fill_value=0, dtype=np.float32)
        self.game_info.terrain_height.data_numpy = height_data

        self.scale = 0.64
        self.px = int(100 * 0.64)
        self.py = int(128 - 1 - (100 * 0.64))

    def run_transform(self, unit=None, effects=None):
        creep_mock = MockGrid(self.map_w, self.map_h, fill_value=0)
        obs = {
            'units': [], 'enemy_units': [], 'effects': effects or [],
            'game_info': self.game_info, 'game_data': None, 'creep': creep_mock
        }
        if unit:
            key = 'units' if unit.alliance == 1 else 'enemy_units'
            obs[key] = [unit]
        return self.transformer.transform(obs)

    def get_val(self, layers, const_name, x=None, y=None):
        if x is None: x = self.px
        if y is None: y = self.py
        return layers[LayerConstants.LAYERS[const_name], y, x]

    def check_unit(self, unit_name, alliance=1, **kwargs):
        u = MockUnit(unit_name, alliance=alliance, **kwargs)
        layers = self.run_transform(unit=u)
        prefix = "allied" if alliance == 1 else "enemy"
        val = self.get_val(layers, f"{prefix}_unit_density")
        self.assertGreater(val, 0.0)
        if kwargs.get('is_biological'):
            self.assertGreater(self.get_val(layers, f"{prefix}_unit_biological"), 0.0)
        if kwargs.get('is_mechanical'):
            self.assertGreater(self.get_val(layers, f"{prefix}_unit_mechanical"), 0.0)
        return layers

    def test_MARINE(self):
        self.check_unit("MARINE", is_light=True, is_biological=True)

    def test_SCV(self):
        # Restored test
        layers = self.check_unit("SCV", is_light=True, is_biological=True, is_mechanical=True)
        self.assertGreater(self.get_val(layers, "allied_unit_biological"), 0.0)
        self.assertGreater(self.get_val(layers, "allied_unit_mechanical"), 0.0)

    def test_INVISIBLE_UNIT(self):
        # Restored test
        u = MockUnit("DARKTEMPLAR", 100, 100, alliance=4, is_cloaked=True, is_detected=False)
        layers = self.run_transform(u)
        self.assertEqual(self.get_val(layers, "enemy_unit_density"), 0.0)

    def test_DAMAGED_UNIT(self):
        # Restored test
        u = MockUnit("MARINE", 100, 100, alliance=1)
        u.health_max = 45
        u.health = 10
        u.shield_max = 0
        u.shield = 0
        layers = self.run_transform(u)
        
        # Expected value is log1p(10/45)
        expected = np.log1p(10.0 / 45.0)
        actual = self.get_val(layers, "allied_health")
        self.assertAlmostEqual(actual, expected, places=4)

    def test_PSIONIC_STORM(self):
        # Restored test
        eff = MockEffect(1, 100, 100)  # ID 1 = Storm
        layers = self.run_transform(effects=[eff])
        val = self.get_val(layers, "enemy_g_dps_base")
        self.assertGreater(val, 0.0)

    def test_LIBERATOR_AG_ZONE(self):
        lib = MockUnit("LIBERATORAG", 50, 50, alliance=1, is_armored=True, is_flying=True)

        # FIX: Explicitly set type_id to a Mock with correct value
        lib.type_id = MockUnitType("LIBERATORAG", 154)

        eff_id = 82  # EffectId.LIBERATORTARGETMORPHDELAYPERSISTENT
        eff = MockEffect(eff_id, 60, 50)

        creep_mock = MockGrid(self.map_w, self.map_h, fill_value=0)
        obs = {
            'units': [lib], 'enemy_units': [], 'effects': [eff],
            'game_info': self.game_info, 'game_data': None,
            'creep': creep_mock
        }
        layers = self.transformer.transform(obs)

        px_zone = int(60 * 0.64)
        py_zone = int(128 - 1 - (50 * 0.64))

        self.assertGreater(layers[LayerConstants.LAYERS["allied_g_dps_base"], py_zone, px_zone], 0.0)

    def test_ORACLE_NO_ENERGY(self):
        u = MockUnit("ORACLE", 100, 100, alliance=4, is_light=True, is_flying=True, energy=0)
        layers = self.run_transform(u)
        val = self.get_val(layers, "enemy_g_dps_base")
        self.assertEqual(val, 0.0)

    def test_INCOMPLETE_DETECTOR(self):
        u = MockUnit("MISSILETURRET", 100, 100, alliance=1, build_progress=0.5)
        layers = self.run_transform(u)
        self.assertGreater(self.get_val(layers, "allied_unit_density"), 0.0)
        self.assertEqual(self.get_val(layers, "allied_detection", x=self.px), 0.0)

    def test_TERRAN_SCAN(self):
        u = MockUnit("SCANNERSWEEP", 100, 100, alliance=1)
        # FIX: Explicitly set type_id
        u.type_id = MockUnitType("SCANNERSWEEP", 170)

        layers = self.run_transform(u)
        self.assertEqual(self.get_val(layers, "allied_detection"), 1.0)
        self.assertEqual(self.get_val(layers, "allied_unit_density"), 0.0)

    def test_FORCE_FIELD(self):
        u = MockUnit("FORCEFIELD", 100, 100, alliance=1)
        # FIX: Explicitly set type_id
        u.type_id = MockUnitType("FORCEFIELD", 135)

        layers = self.run_transform(u)
        path_val = layers[LayerConstants.LAYERS["terrain_pathable"], self.py, self.px]
        self.assertEqual(path_val, 0.0)
        self.assertGreater(self.get_val(layers, "allied_unit_density"), 0.0)

    def test_SENSOR_BLIP(self):
        u = MockUnit("MARINE", 100, 100, alliance=4)
        u.is_blip = True
        layers = self.run_transform(u)
        self.assertGreater(self.get_val(layers, "enemy_unit_density"), 0.0)
        self.assertEqual(self.get_val(layers, "enemy_unit_biological"), 0.0)

    def test_HALLUCINATION_DPS(self):
        u = MockUnit("ARCHON", 100, 100, alliance=4)
        u.is_hallucination = True
        layers = self.run_transform(u)
        self.assertGreater(self.get_val(layers, "enemy_unit_density"), 0.0)
        self.assertEqual(self.get_val(layers, "enemy_g_dps_base"), 0.0)

    def test_WARPING_UNIT_DPS(self):
        u = MockUnit("STALKER", 100, 100, alliance=4, build_progress=0.5)
        layers = self.run_transform(u)
        self.assertGreater(self.get_val(layers, "enemy_unit_density"), 0.0)
        self.assertEqual(self.get_val(layers, "enemy_g_dps_base"), 0.0)


if __name__ == '__main__':
    unittest.main()
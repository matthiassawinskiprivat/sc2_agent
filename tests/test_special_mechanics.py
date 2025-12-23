import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.effect_id import EffectId


# --- MOCK INFRASTRUCTURE ---

class MockPoint2:
    def __init__(self, x, y): self.x, self.y = x, y

    def distance_to_point2(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class MockUnit:
    def __init__(self, type_id, x, y, alliance=1, radius=1.0):
        self.type_id = type_id
        self.position = MockPoint2(x, y)
        self.alliance = alliance
        self.radius = radius
        self.is_structure = False
        self.is_cloaked = False
        self.is_detected = True
        self.is_snapshot = False
        self.is_blip = False
        self.is_hallucination = False
        self.is_ready = True
        self.build_progress = 1.0

        self.health = 100
        self.health_max = 100
        self.shield = 0
        self.shield_max = 0
        self.energy = 0
        self.energy_max = 0

        self.weapon_cooldown = 0
        self.detect_range = 0
        self.attack_upgrade_level = 0

        self.is_biological = False
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.is_flying = False

        self.name = type_id.name if hasattr(type_id, "name") else str(type_id)


class MockEffect:
    def __init__(self, eid, x, y):
        self.id = eid
        self.positions = [MockPoint2(x, y)]


class TestTransformerComprehensive(unittest.TestCase):

    def setUp(self):
        self.map_size = 128
        self.transformer = ObservationTransformer(map_size=(self.map_size, self.map_size))

        self.game_info = MagicMock()
        self.game_info.map_size.width = 128
        self.game_info.map_size.height = 128

        # FIX: Define playable_area
        self.game_info.playable_area.x = 0
        self.game_info.playable_area.y = 0
        self.game_info.playable_area.width = 128
        self.game_info.playable_area.height = 128

        self.game_info.pathing_grid.data_numpy = np.zeros((128, 128))
        self.game_info.terrain_height.data_numpy = np.zeros((128, 128))

        self.transformer.fallback_db = {
            "TEST_GROUND_UNIT": {
                "weapons": [{"type": 1, "damage": 10, "attacks": 1, "period": 1.0, "range": 5, "bonuses": []}]
            },
            "TEST_AIR_UNIT": {
                "weapons": [{"type": 2, "damage": 10, "attacks": 1, "period": 1.0, "range": 5, "bonuses": []}]
            },
            "TEST_VERSATILE_UNIT": {
                "weapons": [{"type": 3, "damage": 10, "attacks": 1, "period": 1.0, "range": 5, "bonuses": []}]
            },
            "TEST_BONUS_UNIT": {
                "weapons": [{"type": 1, "damage": 10, "attacks": 1, "period": 1.0, "range": 5,
                             "bonuses": [{"type": 1, "bonus": 5}, {"type": 2, "bonus": 10}]}]
            }
        }

    def get_layer(self, layers, name):
        return layers[LayerConstants.LAYERS[name]]

    def run_obs(self, units=[], enemy_units=[], effects=[]):
        raw = {
            'units': units,
            'enemy_units': enemy_units,
            'effects': effects,
            'game_info': self.game_info,
            'game_data': MagicMock()
        }
        return self.transformer.transform(raw)

    def test_unit_attributes_mapping(self):
        u = MockUnit(UnitTypeId.MARINE, 64, 64, alliance=1)
        u.is_biological = True
        u.is_mechanical = True
        u.is_armored = True
        u.is_massive = True
        u.is_flying = True

        layers = self.run_obs(units=[u])

        self.assertGreater(np.sum(self.get_layer(layers, "allied_unit_biological")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "allied_unit_mechanical")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "allied_unit_armored")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "allied_unit_massive")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "allied_unit_flying")), 0)

    def test_unit_vitals_normalization(self):
        u = MockUnit(UnitTypeId.MARINE, 64, 64, alliance=1)
        u.health_max = 200
        u.health = 100
        u.energy_max = 100
        u.energy = 25

        layers = self.run_obs(units=[u])

        hp_layer = self.get_layer(layers, "allied_health")
        en_layer = self.get_layer(layers, "allied_energy")

        expected_hp = np.log1p(0.5)
        expected_en = np.log1p(0.25)

        self.assertAlmostEqual(np.max(hp_layer), expected_hp, delta=0.01)
        self.assertAlmostEqual(np.max(en_layer), expected_en, delta=0.01)

    def test_dps_ground_attacker(self):
        mock_id = MagicMock()
        mock_id.name = "TEST_GROUND_UNIT"
        u = MockUnit(mock_id, 64, 64, alliance=2)
        layers = self.run_obs(enemy_units=[u])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_dps_air_attacker(self):
        mock_id = MagicMock()
        mock_id.name = "TEST_AIR_UNIT"
        u = MockUnit(mock_id, 64, 64, alliance=2)
        layers = self.run_obs(enemy_units=[u])

        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_dps_versatile_attacker(self):
        mock_id = MagicMock()
        mock_id.name = "TEST_VERSATILE_UNIT"
        u = MockUnit(mock_id, 64, 64, alliance=2)
        layers = self.run_obs(enemy_units=[u])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_dps_bonus_damage(self):
        mock_id = MagicMock()
        mock_id.name = "TEST_BONUS_UNIT"
        u = MockUnit(mock_id, 64, 64, alliance=2)
        layers = self.run_obs(enemy_units=[u])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_light")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_armor")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_massive")), 0)

    def test_nuke_hits_air_and_ground(self):
        nuke = MockEffect(EffectId.NUKEPERSISTENT, 64, 64)
        layers = self.run_obs(effects=[nuke])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_bile_hits_air_and_ground(self):
        bile = MockEffect(EffectId.RAVAGERCORROSIVEBILECP, 64, 64)
        layers = self.run_obs(effects=[bile])
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_storm_ground_only(self):
        storm = MockEffect(EffectId.PSISTORMPERSISTENT, 64, 64)
        layers = self.run_obs(effects=[storm])
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_scan_effect_detection(self):
        scan = MockEffect(EffectId.SCANNERSWEEP, 64, 64)
        layers = self.run_obs(effects=[scan])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_detection")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)

    def test_structure_footprint(self):
        struct = MockUnit(UnitTypeId.COMMANDCENTER, 64, 64, alliance=1, radius=5.0)
        struct.is_structure = True
        layers = self.run_obs(units=[struct])

        layer = self.get_layer(layers, "allied_unit_density")
        non_zero_pixels = np.count_nonzero(layer)
        self.assertGreater(non_zero_pixels, 10)

    def test_cloaked_enemy_logic(self):
        u1 = MockUnit(UnitTypeId.BANSHEE, 64, 64, alliance=2)
        u1.is_cloaked = True
        u1.is_detected = False
        u1.is_snapshot = False
        layers1 = self.run_obs(enemy_units=[u1])
        self.assertEqual(np.sum(self.get_layer(layers1, "enemy_unit_density")), 0)

        u2 = MockUnit(UnitTypeId.BANSHEE, 64, 64, alliance=2)
        u2.is_cloaked = True
        u2.is_detected = False
        u2.is_snapshot = True
        layers2 = self.run_obs(enemy_units=[u2])
        self.assertGreater(np.sum(self.get_layer(layers2, "enemy_unit_density")), 0)

        u3 = MockUnit(UnitTypeId.BANSHEE, 64, 64, alliance=2)
        u3.is_cloaked = True
        u3.is_detected = True
        layers3 = self.run_obs(enemy_units=[u3])
        self.assertGreater(np.sum(self.get_layer(layers3, "enemy_unit_density")), 0)
        self.assertGreater(np.sum(self.get_layer(layers3, "enemy_unit_cloak")), 0)

    def test_scanner_sweep(self):
        # 170
        scan = MockUnit(170, 50, 50, alliance=1)
        scan.radius = 13.0

        layers = self.run_obs(units=[scan])

        self.assertGreater(np.sum(self.get_layer(layers, "allied_detection")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "allied_unit_density")), 0)

    def test_force_field(self):
        # 135
        ff = MockUnit(135, 40, 40, alliance=2)
        layers = self.run_obs(enemy_units=[ff])
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_unit_density")), 0)

    def test_sensor_blip(self):
        blip = MockUnit(UnitTypeId.MARINE, 30, 30, alliance=2)
        blip.is_blip = True
        blip.is_biological = True

        layers = self.run_obs(enemy_units=[blip])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_unit_density")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_unit_biological")), 0)


if __name__ == '__main__':
    unittest.main()

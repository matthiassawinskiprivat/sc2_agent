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


class MockPoint2:
    def __init__(self, x, y): self.x, self.y = x, y


class MockUnit:
    def __init__(self, type_id, x, y, alliance=1):
        self.type_id = type_id
        self.position = MockPoint2(x, y)
        self.alliance = alliance
        self.radius = 1.0
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
        self.energy_max = 200
        self.weapon_cooldown = 0
        self.detect_range = 0
        self.is_biological = False
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.is_flying = False
        self.attack_upgrade_level = 0
        self.name = type_id.name if hasattr(type_id, "name") else str(type_id)


class MockEffect:
    def __init__(self, eid, x, y):
        self.id = eid
        self.positions = [MockPoint2(x, y)]


class TestCombatLogic(unittest.TestCase):
    def setUp(self):
        self.transformer = ObservationTransformer(map_size=(128, 128))
        self.game_info = MagicMock()
        self.game_info.map_size.width = 128
        self.game_info.map_size.height = 128

        # FIX: Define playable_area with integers to prevent MagicMock comparison error
        self.game_info.playable_area.x = 0
        self.game_info.playable_area.y = 0
        self.game_info.playable_area.width = 128
        self.game_info.playable_area.height = 128

        self.game_info.pathing_grid.data_numpy = np.zeros((128, 128))
        self.game_info.terrain_height.data_numpy = np.zeros((128, 128))

    def get_layer(self, layers, name):
        return layers[LayerConstants.LAYERS[name]]

    def transform(self, units=[], enemy_units=[], effects=[]):
        raw = {
            'units': units,
            'enemy_units': enemy_units,
            'effects': effects,
            'game_info': self.game_info,
            'game_data': MagicMock()
        }
        return self.transformer.transform(raw)

    def test_dps_hallucination_exclusion(self):
        u = MockUnit(UnitTypeId.COLOSSUS, 50, 50, alliance=2)
        u.is_hallucination = True
        layers = self.transform([], [u])
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)

    def test_oracle_energy_dps(self):
        u_low = MockUnit(UnitTypeId.ORACLE, 50, 50, alliance=2)
        u_low.energy = 10
        layers_low = self.transform([], [u_low])
        self.assertEqual(np.sum(self.get_layer(layers_low, "enemy_g_dps_light")), 0)

        u_high = MockUnit(UnitTypeId.ORACLE, 50, 50, alliance=2)
        u_high.energy = 50
        layers_high = self.transform([], [u_high])
        self.assertGreater(np.sum(self.get_layer(layers_high, "enemy_g_dps_light")), 0)

    def test_widow_mine_burrowed(self):
        u_up = MockUnit(UnitTypeId.WIDOWMINE, 50, 50, alliance=2)
        layers_up = self.transform([], [u_up])

        u_down = MockUnit(UnitTypeId.WIDOWMINEBURROWED, 60, 60, alliance=2)
        layers_down = self.transform([], [u_down])
        self.assertGreater(np.sum(self.get_layer(layers_down, "enemy_g_dps_base")), 0)

    def test_healing_energy_thresholds(self):
        m_low = MockUnit(UnitTypeId.MEDIVAC, 50, 50, alliance=1)
        m_low.energy = 2
        layers_low = self.transform([m_low])
        self.assertEqual(np.sum(self.get_layer(layers_low, "allied_hps_field")), 0)

        m_high = MockUnit(UnitTypeId.MEDIVAC, 60, 60, alliance=1)
        m_high.energy = 50
        layers_high = self.transform([m_high])
        self.assertGreater(np.sum(self.get_layer(layers_high, "allied_hps_field")), 0)

    def test_liberator_zone_ownership(self):
        eid = EffectId.LIBERATORTARGETMORPHDELAYPERSISTENT
        eff = MockEffect(eid, 50, 50)

        lib_ally = MockUnit(UnitTypeId.LIBERATORAG, 55, 50, alliance=1)  # dist 5
        lib_enemy = MockUnit(UnitTypeId.LIBERATORAG, 80, 50, alliance=2)  # dist 30

        layers = self.transform([lib_ally], [lib_enemy], [eff])
        self.assertGreater(np.sum(self.get_layer(layers, "allied_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)

        lib_ally.position = MockPoint2(20, 50)
        lib_enemy.position = MockPoint2(52, 50)

        layers = self.transform([lib_ally], [lib_enemy], [eff])
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "allied_g_dps_base")), 0)

    def test_nuke_effect(self):
        eid = EffectId.NUKEPERSISTENT
        eff = MockEffect(eid, 50, 50)
        layers = self.transform([], [], [eff])
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)


if __name__ == '__main__':
    unittest.main()
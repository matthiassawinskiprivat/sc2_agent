import unittest
import sys
import os
import json
import numpy as np
from unittest.mock import MagicMock, PropertyMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


# --- MOCK CLASSES (Simulate SC2 Library) ---
class MockPoint2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to_point2(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class MockUnitType:
    def __init__(self, name, value=0):
        self.name = name
        self.value = value


class MockUnit:
    def __init__(self, name, x, y, alliance=1, health=100, max_health=100,
                 cooldown=0, flying=False, light=False, armored=False,
                 cloaked=False, detected=True, energy=0, tag=0):
        self.type_id = MockUnitType(name)
        self.position = MockPoint2(x, y)
        self.alliance = alliance  # 1=Self, 4=Enemy
        self.health = health
        self.health_max = max_health
        self.shield = 0
        self.shield_max = 0
        self.weapon_cooldown = cooldown
        self.is_flying = flying
        self.is_light = light
        self.is_armored = armored
        self.is_cloaked = cloaked
        self.is_detected = detected
        self.energy = energy
        self.energy_max = 200
        self.radius = 0.5
        self.detect_range = 0
        self.attack_upgrade_level = 0
        self.tag = tag
        self.orders = []


class MockEffect:
    def __init__(self, id_val, x, y):
        self.id = MagicMock()
        self.id.value = id_val  # support .value access
        # Also allow direct integer comparison if logic changes
        self.id.__eq__ = lambda s, o: id_val == o
        self.positions = [MockPoint2(x, y)]


# --- THE TEST SUITE ---
class TestObservationTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure we can find the unit data
        # Assumes tests/ is one level deep in project root
        cls.data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'unit_data.json')
        if not os.path.exists(cls.data_path):
            # Fallback for different running contexts
            cls.data_path = os.path.join(os.getcwd(), 'data', 'unit_data.json')

        if not os.path.exists(cls.data_path):
            print(
                f"WARNING: Could not find unit_data.json at {cls.data_path}. Tests relying on specific unit stats might fail or use fallback defaults.")

    def setUp(self):
        # Initialize Transformer with 128x128 map
        # Note: SC2 coords 0-200 map to 0-128 pixels
        self.transformer = ObservationTransformer(map_size=(128, 128), unit_data_file=self.data_path)
        self.map_w = 200  # Standard SC2 map width for norm
        self.map_h = 200

        # Create a dummy GameInfo
        self.mock_game_info = MagicMock()
        self.mock_game_info.map_size.width = self.map_w
        self.mock_game_info.map_size.height = self.map_h
        # Mock terrain grids
        self.mock_game_info.terrain_height = np.zeros((200, 200))
        self.mock_game_info.pathing_grid = MagicMock()
        self.mock_game_info.pathing_grid.width = 200
        self.mock_game_info.pathing_grid.height = 200
        self.mock_game_info.pathing_grid.is_pathable.return_value = True

    def _get_pixel_coords(self, game_x, game_y):
        """Helper to convert game coords to pixel coords expected by transformer"""
        px = int(game_x / self.map_w * 128)
        py = 128 - 1 - int(game_y / self.map_h * 128)
        return px, py

    def test_layer_indices(self):
        """Verify LayerConstants are correctly mapped"""
        self.assertEqual(LayerConstants.get_num_layers(), 23)
        self.assertEqual(LayerConstants.LAYERS[LayerConstants.ALLIED_UNIT_DENSITY], 0)
        # Fixed syntax errors here:
        self.assertEqual(LayerConstants.LAYERS[LayerConstants.ENEMY_UNIT_DENSITY], 4)
        self.assertEqual(LayerConstants.LAYERS[LayerConstants.ALLIED_DPS_GROUND_LIGHT], 10)

    # --- UNIT STATE TESTS ---

    def test_allied_marine_state(self):
        """Test Layer 0 (Density) and Layer 1 (Attribute) for Marine"""
        # Marine: Light, Ground
        marine = MockUnit("MARINE", 100, 100, alliance=1, light=True)

        obs = {'units': [marine], 'enemy_units': [], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(100, 100)

        # Layer 0: Density (Should be 1.0 as cooldown is 0)
        self.assertEqual(layers[0, py, px], 1.0, "Marine Density should be 1.0")

        # Layer 1: Attribute (Light = 0.33)
        self.assertAlmostEqual(layers[1, py, px], 0.33, places=2, msg="Marine should be Light (0.33)")

        # Layer 2: Flying (Should be 0)
        self.assertEqual(layers[2, py, px], 0.0)

    def test_enemy_marauder_state(self):
        """Test Layer 4 (Enemy Density) and Layer 5 (Attribute) for Marauder"""
        # Marauder: Armored, Ground
        marauder = MockUnit("MARAUDER", 50, 50, alliance=4, armored=True)

        obs = {'units': [], 'enemy_units': [marauder], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(50, 50)

        # Layer 4: Enemy Density
        self.assertEqual(layers[4, py, px], 1.0)

        # Layer 5: Attribute (Armored = 0.66)
        self.assertAlmostEqual(layers[5, py, px], 0.66, places=2, msg="Marauder should be Armored (0.66)")

    def test_flying_unit(self):
        """Test Flying Layers for Mutalisk"""
        muta = MockUnit("MUTALISK", 100, 100, alliance=1, flying=True, light=True)
        obs = {'units': [muta], 'enemy_units': [], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(100, 100)
        self.assertEqual(layers[2, py, px], 1.0, "Allied Flying Layer should be 1.0")

    # --- DPS & THREAT TESTS ---

    def test_marine_dps_output(self):
        """
        Verify Marine generates DPS on Allied Layers.
        Marine (Base Dmg 6, Cooldown ~0.61) -> ~9.8 DPS.
        Targets: Ground & Air.
        Should appear on ALL Allied DPS layers (10-13) because it hits everything.
        """
        marine = MockUnit("MARINE", 100, 100, alliance=1)
        obs = {'units': [marine], 'enemy_units': [], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(100, 100)

        # Check Allied Ground Light (Layer 10)
        # Use > 0 check first
        self.assertGreater(layers[10, py, px], 0.0, "Marine should deal damage to Ground Light")

        # Check Air Layers (Marine shoots up)
        self.assertGreater(layers[12, py, px], 0.0, "Marine should deal damage to Air Light")

    def test_marauder_bonus_dps(self):
        """
        Verify Marauder deals MORE damage to Armored than Light.
        Marauder: Base 10, Bonus +10 vs Armored.
        """
        marauder = MockUnit("MARAUDER", 100, 100, alliance=4, armored=True)  # Enemy
        obs = {'units': [], 'enemy_units': [marauder], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(105, 100)  # Check a pixel in range

        # Enemy Ground Light (Layer 14) vs Enemy Ground Armored (Layer 15)
        dps_light = layers[14, py, px]
        dps_armor = layers[15, py, px]

        self.assertGreater(dps_light, 0.0, "Should have base damage")
        self.assertGreater(dps_armor, dps_light, "Armored DPS should be higher than Light DPS for Marauder")

    def test_colossus_anti_light(self):
        """Verify Colossus deals bonus vs Light"""
        colossus = MockUnit("COLOSSUS", 100, 100, alliance=4)
        obs = {'units': [], 'enemy_units': [colossus], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(105, 100)

        dps_light = layers[14, py, px]
        dps_armor = layers[15, py, px]

        self.assertGreater(dps_light, dps_armor, "Colossus should deal more to Light than Armored")

    def test_tempest_range_split(self):
        """Verify Tempest has different ranges for Ground (10) vs Air (14)"""
        tempest = MockUnit("TEMPEST", 100, 100, alliance=4, flying=True)
        obs = {'units': [], 'enemy_units': [tempest], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        # Check Short Range (Ground)
        px_near, py_near = self._get_pixel_coords(109, 100)  # Range 9
        self.assertGreater(layers[14, py_near, px_near], 0.0, "Should hit Ground at range 9")

        # Check Long Range (Air Only)
        px_far, py_far = self._get_pixel_coords(113, 100)  # Range 13
        self.assertEqual(layers[14, py_far, px_far], 0.0, "Should NOT hit Ground at range 13")
        self.assertGreater(layers[16, py_far, px_far], 0.0, "Should hit Air at range 13")

    # --- SPECIAL UNITS & OVERRIDES ---

    def test_oracle_energy_logic(self):
        """Oracle should only threaten if Energy >= 25"""
        # Case A: Low Energy
        oracle_empty = MockUnit("ORACLE", 50, 50, alliance=4, energy=10)
        # Case B: High Energy
        oracle_full = MockUnit("ORACLE", 150, 150, alliance=4, energy=50)

        obs = {'units': [], 'enemy_units': [oracle_empty, oracle_full], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px_a, py_a = self._get_pixel_coords(50, 50)
        px_b, py_b = self._get_pixel_coords(150, 150)

        # Check Enemy Ground DPS
        self.assertEqual(layers[14, py_a, px_a], 0.0, "Empty Oracle should have 0 DPS")
        self.assertGreater(layers[14, py_b, px_b], 0.0, "Full Oracle should have DPS")

    def test_widow_mine_cooldown(self):
        """Widow Mine threat should scale with cooldown"""
        # Mine A: Ready
        mine_ready = MockUnit("WIDOWMINEBURROWED", 50, 50, alliance=4, cooldown=0)
        # Mine B: Just Fired (High cooldown)
        mine_fired = MockUnit("WIDOWMINEBURROWED", 150, 150, alliance=4, cooldown=600)  # ~27s

        obs = {'units': [], 'enemy_units': [mine_ready, mine_fired], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px_a, py_a = self._get_pixel_coords(50, 50)
        px_b, py_b = self._get_pixel_coords(150, 150)

        threat_ready = layers[14, py_a, px_a]  # Enemy Ground Light
        threat_fired = layers[14, py_b, px_b]

        self.assertGreater(threat_ready, 0.8, "Ready Mine should be high threat")
        self.assertLess(threat_fired, 0.2, "Fired Mine should be low threat")

    def test_liberator_zone_logic(self):
        """Test LiberatorAG + Effect Logic"""
        # 1. Create Allied Liberator Unit
        lib = MockUnit("LIBERATORAG", 50, 50, alliance=1)

        # 2. Create Effect (ID 82 or 83) at target location (60, 50)
        # Note: Effect is separate from unit position
        eff = MockEffect(83, 60, 50)

        obs = {
            'units': [lib],
            'enemy_units': [],
            'effects': [eff],
            'game_info': self.mock_game_info
        }

        layers = self.transformer.transform(obs)

        px_unit, py_unit = self._get_pixel_coords(50, 50)
        px_zone, py_zone = self._get_pixel_coords(60, 50)

        # Check Layer 10 (Allied Ground Light DPS)
        # Should have DPS at Zone, NOT at Unit
        self.assertEqual(layers[10, py_unit, px_unit], 0.0, "Liberator Unit itself should have 0 DPS radius")
        self.assertGreater(layers[10, py_zone, px_zone], 0.0, "Liberator Effect Zone should have high DPS")

    # --- CLOAK & DETECTION ---

    def test_enemy_cloak_visibility(self):
        """Cloaked Enemy without detection should not appear on Body Layer"""
        # Cloaked + Not Detected
        dt = MockUnit("DARKTEMPLAR", 50, 50, alliance=4, cloaked=True, detected=False)

        obs = {'units': [], 'enemy_units': [dt], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(50, 50)

        # Layer 4: Enemy Density
        self.assertEqual(layers[4, py, px], 0.0, "Undetected unit should be invisible")

    def test_detection_circle(self):
        """Test Detection Layer (18/19)"""
        observer = MockUnit("OBSERVER", 100, 100, alliance=1)
        observer.detect_range = 11.0

        obs = {'units': [observer], 'enemy_units': [], 'game_info': self.mock_game_info}
        layers = self.transformer.transform(obs)

        px, py = self._get_pixel_coords(100, 100)

        # Layer 18: Allied Detection
        self.assertEqual(layers[18, py, px], 1.0, "Detection circle should be present")


if __name__ == '__main__':
    unittest.main()
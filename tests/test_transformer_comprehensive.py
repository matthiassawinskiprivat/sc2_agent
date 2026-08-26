import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants
from tests.obs_transformer_test_utils import get_layer, transform_map
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

        # Vitals
        self.health = 100
        self.health_max = 100
        self.shield = 0
        self.shield_max = 0
        self.energy = 0
        self.energy_max = 0

        # Cooldowns & Range
        self.weapon_cooldown = 0
        self.detect_range = 0
        self.attack_upgrade_level = 0

        # Attributes
        self.is_biological = False
        self.is_mechanical = False
        self.is_armored = False
        self.is_massive = False
        self.is_flying = False

        # Name for DB lookup
        self.name = type_id.name if hasattr(type_id, "name") else str(type_id)


class MockEffect:
    def __init__(self, eid, x, y):
        self.id = eid
        self.positions = [MockPoint2(x, y)]


class TestTransformerComprehensive(unittest.TestCase):

    def setUp(self):
        # 1. Setup Transformer
        self.map_size = 128
        self.transformer = ObservationTransformer(map_size=(self.map_size, self.map_size))

        # 2. Setup Mock GameInfo (Clean Slate)
        self.game_info = MagicMock()
        self.game_info.map_size.width = 128
        self.game_info.map_size.height = 128

        # FIX: Explicitly create playable_area with integer values
        self.game_info.playable_area.x = 0
        self.game_info.playable_area.y = 0
        self.game_info.playable_area.width = 128
        self.game_info.playable_area.height = 128

        self.game_info.pathing_grid.data_numpy = np.zeros((128, 128))
        self.game_info.terrain_height.data_numpy = np.zeros((128, 128))

        # 3. Inject Custom "Unit Database"
        # This decouples tests from external JSON files. We test the LOGIC, not the data.
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
                # Type 1=Light, 2=Armored
            }
        }

    def get_layer(self, layers, name):
        return get_layer(layers, name)

    def run_obs(self, units=[], enemy_units=[], effects=[]):
        raw = {
            'units': units,
            'enemy_units': enemy_units,
            'effects': effects,
            'game_info': self.game_info,
            'game_data': MagicMock()
        }
        return transform_map(self.transformer, raw)

    # ==============================================================================
    # 1. UNIT ATTRIBUTE TESTS
    # ==============================================================================

    def test_unit_attributes_mapping(self):
        """Verify Bio, Mech, Armored, Massive, Flying flags appear on correct layers."""
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
        """Verify Health and Energy are normalized correctly (Log1p scale)."""
        u = MockUnit(UnitTypeId.MARINE, 64, 64, alliance=1)
        u.health_max = 200
        u.health = 100  # 50% ratio
        u.energy_max = 100
        u.energy = 25  # 25% ratio

        layers = self.run_obs(units=[u])

        # Check center pixel
        hp_layer = self.get_layer(layers, "allied_health")
        en_layer = self.get_layer(layers, "allied_energy")

        # Expected: health = damage fraction; energy still log1p(ratio)
        expected_hp = 0.5
        expected_en = np.log1p(0.25)

        self.assertAlmostEqual(np.max(hp_layer), expected_hp, delta=0.01)
        self.assertAlmostEqual(np.max(en_layer), expected_en, delta=0.01)

    # ==============================================================================
    # 2. DPS PROJECTION TESTS (Logic Verification)
    # ==============================================================================

    def test_dps_ground_attacker(self):
        """A unit targeting ONLY Ground should appear on G_DPS but NOT A_DPS."""
        # Use custom type name hooked in setUp
        mock_id = MagicMock()
        mock_id.name = "TEST_GROUND_UNIT"

        u = MockUnit(mock_id, 64, 64, alliance=2)  # Enemy

        layers = self.run_obs(enemy_units=[u])

        # Check Enemy Ground DPS
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        # Check Enemy Air DPS -> Should be ZERO
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_dps_air_attacker(self):
        """A unit targeting ONLY Air should appear on A_DPS but NOT G_DPS."""
        mock_id = MagicMock()
        mock_id.name = "TEST_AIR_UNIT"

        u = MockUnit(mock_id, 64, 64, alliance=2)

        layers = self.run_obs(enemy_units=[u])

        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_dps_versatile_attacker(self):
        """A unit targeting BOTH should appear on BOTH layers."""
        mock_id = MagicMock()
        mock_id.name = "TEST_VERSATILE_UNIT"

        u = MockUnit(mock_id, 64, 64, alliance=2)

        layers = self.run_obs(enemy_units=[u])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)

    def test_dps_bonus_damage(self):
        """Verify bonus damage vs Light/Armored is applied to specific layers."""
        mock_id = MagicMock()
        mock_id.name = "TEST_BONUS_UNIT"

        u = MockUnit(mock_id, 64, 64, alliance=2)

        layers = self.run_obs(enemy_units=[u])

        # Base
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        # Light Bonus
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_light")), 0)
        # Armor Bonus
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_armor")), 0)
        # Massive Bonus (Should be 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_massive")), 0)

    # ==============================================================================
    # 3. EFFECT THREAT TESTS (Critical Regressions)
    # ==============================================================================

    def test_nuke_hits_air_and_ground(self):
        """CRITICAL: Verify Nukes project threat to BOTH Ground and Air layers."""
        nuke = MockEffect(EffectId.NUKEPERSISTENT, 64, 64)

        layers = self.run_obs(effects=[nuke])

        # Must hit Ground
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0, "Nuke missing from Ground DPS")
        # Must hit Air (This was the bug!)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0, "Nuke missing from Air DPS")
        self.assertEqual(np.sum(self.get_layer(layers, "allied_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "allied_a_dps_base")), 0)

    def test_bile_hits_air_and_ground(self):
        """CRITICAL: Verify Corrosive Bile hits BOTH."""
        bile = MockEffect(EffectId.RAVAGERCORROSIVEBILECP, 64, 64)

        layers = self.run_obs(effects=[bile])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "allied_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "allied_a_dps_base")), 0)

    def test_storm_ground_only(self):
        """Psionic Storm should usually hit Ground only (standard ladder behavior)."""
        storm = MockEffect(EffectId.PSISTORMPERSISTENT, 64, 64)

        layers = self.run_obs(effects=[storm])

        self.assertGreater(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_a_dps_base")), 0)
        self.assertEqual(np.sum(self.get_layer(layers, "allied_g_dps_base")), 0)

    def test_scan_effect_detection(self):
        """Scanner Sweep EFFECT should show as ENEMY Detection."""
        # Note: We use the Effect here (ID 6), not the Unit (ID 170).
        scan = MockEffect(EffectId.SCANNERSWEEP, 64, 64)

        layers = self.run_obs(effects=[scan])

        # Scan Effect -> Enemy Detection Layer
        self.assertGreater(np.sum(self.get_layer(layers, "enemy_detection")), 0)

        # Scan Effect -> Should NOT be DPS
        self.assertEqual(np.sum(self.get_layer(layers, "enemy_g_dps_base")), 0)

    # ==============================================================================
    # 4. STRUCTURE TESTS
    # ==============================================================================

    def test_structure_footprint(self):
        """Ensure structures are drawn with a radius > 1 pixel."""
        # Use a large radius to be sure
        struct = MockUnit(UnitTypeId.COMMANDCENTER, 64, 64, alliance=1, radius=5.0)
        struct.is_structure = True

        layers = self.run_obs(units=[struct])

        layer = self.get_layer(layers, "allied_unit_density")
        non_zero_pixels = np.count_nonzero(layer)

        # A radius 5 circle (scaled 1.0) has area ~ pi*r^2 ~ 78 pixels
        # A single pixel unit has 1.
        self.assertGreater(non_zero_pixels, 10, "Structure footprint is too small (likely 1px)")


if __name__ == '__main__':
    unittest.main()
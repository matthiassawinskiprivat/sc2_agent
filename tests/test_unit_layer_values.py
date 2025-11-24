import unittest
import sys
import os
import json
import numpy as np
from unittest.mock import MagicMock

# --- FIX 1: Add this import so UnitTypeId is defined ---
from sc2.ids.unit_typeid import UnitTypeId

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformers.observation_transformer import ObservationTransformer
from src.utils.layer_constants import LayerConstants


# --- MOCK CLASSES ---
class MockPoint2:
    def __init__(self, x, y): self.x, self.y = x, y

    def distance_to_point2(self, p): return ((self.x - p.x) ** 2 + (self.y - p.y) ** 2) ** 0.5

    def __getitem__(self, item): return self.x if item == 0 else self.y


class MockUnitType:
    def __init__(self, name): self.name = name

    @property
    def value(self): return 0


class MockUnit:
    def __init__(self, name, x=100, y=100, alliance=1,
                 is_light=False, is_armored=False, is_biological=False, is_mechanical=False,
                 is_massive=False, is_psionic=False,
                 is_flying=False, is_cloaked=False, is_detected=True,
                 energy=0, cooldown=0, detect_range=0, build_progress=1.0):

        # --- FIX 2: Use Real Enum for Identity Checks ---
        try:
            self.type_id = UnitTypeId[name]
        except KeyError:
            # Fallback only if the name isn't a valid SC2 unit ID
            self.type_id = MockUnitType(name)

        self.position = MockPoint2(x, y)
        self.alliance = alliance

        # Attributes
        self.is_light = is_light
        self.is_armored = is_armored
        self.is_biological = is_biological
        self.is_mechanical = is_mechanical
        self.is_massive = is_massive
        self.is_psionic = is_psionic
        self.is_flying = is_flying
        self.is_cloaked = is_cloaked
        self.is_detected = is_detected

        # Vitals
        self.health_max = 100
        self.shield_max = 50
        self.health = 100
        self.shield = 50
        self.energy = energy
        self.energy_max = 200 if energy > 0 else 0

        self.weapon_cooldown = cooldown
        self.detect_range = detect_range
        self.radius = 0.5
        self.attack_upgrade_level = 0
        self.build_progress = build_progress
        self.orders = []

        # Dummy property for robustness
        self.is_structure = False
        self.is_ready = (build_progress == 1.0)


class MockEffect:
    def __init__(self, id_val, x, y):
        self.id = MagicMock()
        self.id.value = id_val
        self.id.__eq__ = lambda s, o: id_val == o
        self.positions = [MockPoint2(x, y)]


# --- MAIN TEST SUITE ---
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
        self.game_info.map_size = MagicMock()
        type(self.game_info.map_size).width = 200
        type(self.game_info.map_size).height = 200
        self.game_info.terrain_height = np.zeros((200, 200))

        # --- FIX 3: Assign explicit integers to mock pathing grid ---
        self.game_info.pathing_grid = MagicMock()
        self.game_info.pathing_grid.width = 200
        self.game_info.pathing_grid.height = 200
        self.game_info.pathing_grid.is_pathable.return_value = True
        self.game_info.pathing_grid.__getitem__ = MagicMock(return_value=1)  # [x,y] access

        self.px = int(100 / 200 * 128)
        self.py = 128 - 1 - int(100 / 200 * 128)

    def run_transform(self, unit=None, effects=None):
        obs = {
            'units': [], 'enemy_units': [], 'effects': effects or [],
            'game_info': self.game_info, 'game_data': None,
            'creep': MagicMock()  # Mock creep grid
        }
        # Mock creep grid behavior
        obs['creep'].width = 200
        obs['creep'].height = 200
        obs['creep'].__getitem__ = MagicMock(return_value=0)

        if unit:
            key = 'units' if unit.alliance == 1 else 'enemy_units'
            obs[key] = [unit]

        return self.transformer.transform(obs)

    def get_val(self, layers, const_name, x=None, y=None):
        if x is None: x = self.px
        if y is None: y = self.py
        return layers[LayerConstants.LAYERS[const_name], y, x]

    # --- HELPER: AUTOMATED UNIT CHECKER ---
    def check_unit(self, unit_name, alliance=1, **kwargs):
        """
        Generic test method that validates a unit against the DB and Layer Rules.
        """
        # 1. Create Unit
        u = MockUnit(unit_name, alliance=alliance, **kwargs)
        layers = self.run_transform(unit=u)

        prefix = "allied" if alliance == 1 else "enemy"
        opp_prefix = "enemy" if alliance == 1 else "allied"

        # 2. State Checks
        # Density
        self.assertEqual(self.get_val(layers, f"{prefix}_unit_density"), 1.0 if alliance == 4 else 1.0)
        # Health
        self.assertEqual(self.get_val(layers, f"{prefix}_health"), 1.0)

        # 3. DPS Checks (Dynamic from DB)
        # We calculate what the DPS *should* be based on the DB loaded by transformer
        db_data = self.transformer._get_unit_data_from_json(unit_name)

        # Skip DPS check if it's a special unit handled by overrides or exclusions
        if unit_name in ["LIBERATORAG", "CARRIER", "LURKERMP", "MEDIVAC", "WARPPRISM", "OBSERVER", "OVERLORD",
                         "OVERSEER"]:
            return layers

        if db_data and db_data.get('weapons'):
            for w in db_data['weapons']:
                d_base = (w['damage'] * w['attacks']) / (w['period'] or 1.0)
                # Bonuses
                b_l, b_a, b_m = 0.0, 0.0, 0.0
                for b in w.get('bonuses', []):
                    val = (b['bonus'] * w['attacks']) / (w['period'] or 1.0)
                    if b['type'] == 1: b_l += val
                    if b['type'] == 2: b_a += val
                    if b['type'] == 7: b_m += val

                typ = w['type']  # 1=G, 2=A, 3=Any

                # Check Ground Layers
                if typ == 1 or typ == 3:
                    self._assert_dps(layers, f"{prefix}_g_dps_base", d_base)
                    self._assert_dps(layers, f"{prefix}_g_dps_light", b_l)
                    self._assert_dps(layers, f"{prefix}_g_dps_armor", b_a)
                    self._assert_dps(layers, f"{prefix}_g_dps_massive", b_m)

                # Check Air Layers
                if typ == 2 or typ == 3:
                    self._assert_dps(layers, f"{prefix}_a_dps_base", d_base)
                    self._assert_dps(layers, f"{prefix}_a_dps_light", b_l)
                    self._assert_dps(layers, f"{prefix}_a_dps_armor", b_a)
                    self._assert_dps(layers, f"{prefix}_a_dps_massive", b_m)

        # 4. Verify NO bleed to opponent layers
        self.assertEqual(self.get_val(layers, f"{opp_prefix}_unit_density"), 0.0)
        self.assertEqual(self.get_val(layers, f"{opp_prefix}_health"), 0.0)

        return layers

    def _assert_dps(self, layers, layer_name, raw_dps):
        val = self.get_val(layers, layer_name)
        if raw_dps > 0:
            expected = np.tanh(raw_dps / 50.0)
            self.assertAlmostEqual(val, expected, places=4, msg=f"{layer_name} mismatch for {raw_dps} DPS")
        else:
            self.assertEqual(val, 0.0, f"{layer_name} should be 0")

    # ==================================================================
    # TERRAN TESTS
    # ==================================================================

    def test_SCV(self):
        self.check_unit("SCV", is_light=True, is_biological=True, is_mechanical=True)

    def test_MARINE(self):
        self.check_unit("MARINE", is_light=True, is_biological=True)

    def test_MARAUDER(self):
        # Bonus vs Armor
        self.check_unit("MARAUDER", is_armored=True, is_biological=True)

    def test_SIEGETANK(self):
        self.check_unit("SIEGETANK", is_armored=True, is_mechanical=True)

    def test_SIEGETANKSIEGED(self):
        self.check_unit("SIEGETANKSIEGED", is_armored=True, is_mechanical=True)

    def test_HELLION(self):
        # Bonus vs Light
        self.check_unit("HELLION", is_light=True, is_mechanical=True)

    def test_HELLIONTANK(self):
        # Hellbat
        self.check_unit("HELLIONTANK", is_light=True, is_biological=True, is_mechanical=True)

    def test_CYCLONE(self):
        self.check_unit("CYCLONE", is_mechanical=True)

    def test_THOR(self):
        # Multi-weapon check handled by generic checker
        self.check_unit("THOR", is_armored=True, is_mechanical=True, is_massive=True)

    def test_VIKINGFIGHTER(self):
        # Air Only
        self.check_unit("VIKINGFIGHTER", is_armored=True, is_mechanical=True, is_flying=True)

    def test_VIKINGASSAULT(self):
        # Ground Only
        self.check_unit("VIKINGASSAULT", is_armored=True, is_mechanical=True)

    def test_MEDIVAC(self):
        # Healer - Special Logic
        u = MockUnit("MEDIVAC", 100, 100, alliance=1, is_armored=True, is_flying=True, energy=50)
        layers = self.run_transform(u)
        # HPS Layer
        self.assertGreater(self.get_val(layers, "allied_hps_field"), 0.0)
        # No DPS
        self.assertEqual(self.get_val(layers, "allied_g_dps_base"), 0.0)

    def test_BANSHEE(self):
        self.check_unit("BANSHEE", is_light=True, is_mechanical=True, is_flying=True)

    def test_BATTLECRUISER(self):
        self.check_unit("BATTLECRUISER", is_armored=True, is_mechanical=True, is_massive=True, is_flying=True)

    def test_LIBERATOR(self):
        # Anti-Air
        self.check_unit("LIBERATOR", is_armored=True, is_mechanical=True, is_flying=True)

    def test_RAVEN(self):
        # Detection
        u = MockUnit("RAVEN", 100, 100, alliance=1, is_light=True, is_flying=True, detect_range=11)
        layers = self.run_transform(u)
        self.assertEqual(self.get_val(layers, "allied_detection", x=self.px + 5), 1.0)

    # ==================================================================
    # ZERG TESTS
    # ==================================================================

    def test_ZERGLING(self):
        self.check_unit("ZERGLING", is_light=True, is_biological=True)

    def test_ROACH(self):
        self.check_unit("ROACH", is_armored=True, is_biological=True)

    def test_HYDRALISK(self):
        self.check_unit("HYDRALISK", is_light=True, is_biological=True)

    def test_MUTALISK(self):
        self.check_unit("MUTALISK", is_light=True, is_biological=True, is_flying=True)

    def test_CORRUPTOR(self):
        self.check_unit("CORRUPTOR", is_armored=True, is_biological=True, is_flying=True)

    def test_BROODLORD(self):
        self.check_unit("BROODLORD", is_armored=True, is_biological=True, is_massive=True, is_flying=True)

    def test_ULTRALISK(self):
        self.check_unit("ULTRALISK", is_armored=True, is_biological=True, is_massive=True)

    def test_QUEEN(self):
        # Healer + DPS
        u = MockUnit("QUEEN", 100, 100, alliance=1, is_biological=True, is_psionic=True, energy=100)
        layers = self.run_transform(u)
        # Check DPS (Queen has weapons)
        self.assertGreater(self.get_val(layers, "allied_g_dps_base"), 0.0)
        # Check HPS (Transfuse virtual)
        self.assertGreater(self.get_val(layers, "allied_hps_field"), 0.0)

    # ==================================================================
    # PROTOSS TESTS
    # ==================================================================

    def test_ZEALOT(self):
        self.check_unit("ZEALOT", is_light=True, is_biological=True)

    def test_STALKER(self):
        self.check_unit("STALKER", is_armored=True, is_mechanical=True)

    def test_ADEPT(self):
        self.check_unit("ADEPT", is_light=True, is_biological=True)

    def test_IMMORTAL(self):
        self.check_unit("IMMORTAL", is_armored=True, is_mechanical=True)

    def test_COLOSSUS(self):
        # Hybrid Flying
        u = MockUnit("COLOSSUS", 100, 100, alliance=1, is_armored=True, is_mechanical=True, is_massive=True)
        layers = self.run_transform(u)
        self.assertEqual(self.get_val(layers, "allied_unit_flying"), 0.5)
        self.assertGreater(self.get_val(layers, "allied_g_dps_base"), 0.0)

    def test_VOIDRAY(self):
        self.check_unit("VOIDRAY", is_armored=True, is_mechanical=True, is_flying=True)

    def test_PHOENIX(self):
        self.check_unit("PHOENIX", is_light=True, is_mechanical=True, is_flying=True)

    def test_CARRIER(self):
        # No DPS from body
        u = MockUnit("CARRIER", 100, 100, alliance=1, is_armored=True, is_mechanical=True, is_massive=True,
                     is_flying=True)
        layers = self.run_transform(u)
        self.assertEqual(self.get_val(layers, "allied_g_dps_base"), 0.0)

    def test_INTERCEPTOR(self):
        self.check_unit("INTERCEPTOR", is_light=True, is_mechanical=True, is_flying=True)

    def test_TEMPEST(self):
        # Virtual Override test
        u = MockUnit("TEMPEST", 100, 100, alliance=1, is_armored=True, is_mechanical=True, is_massive=True,
                     is_flying=True)
        layers = self.run_transform(u)
        # Should have Ground AND Air DPS
        self.assertGreater(self.get_val(layers, "allied_g_dps_base"), 0.0)
        self.assertGreater(self.get_val(layers, "allied_a_dps_base"), 0.0)

    # ==================================================================
    # SPECIAL CASES
    # ==================================================================

    def test_WIDOWMINE_BURROWED(self):
        # Ready -> High Virtual DPS
        u = MockUnit("WIDOWMINEBURROWED", 100, 100, alliance=4, is_light=True, is_mechanical=True,
                     is_cloaked=True, is_detected=True, cooldown=0)
        layers = self.run_transform(u)

        # Check Enemy Threat
        val = self.get_val(layers, "enemy_g_dps_base")
        self.assertGreater(val, 0.9, "Ready Mine should be lethal")

        # Check Cloak
        # FIX: Use assertAlmostEqual because layers are float32 and Python floats are float64
        self.assertAlmostEqual(self.get_val(layers, "enemy_unit_cloak"), 0.6, places=5)

    def test_ORACLE_ENERGY(self):
        # High Energy -> DPS
        u = MockUnit("ORACLE", 100, 100, alliance=4, is_light=True, is_flying=True, energy=50)
        layers = self.run_transform(u)
        self.assertGreater(self.get_val(layers, "enemy_g_dps_base"), 0.0)

    def test_LIBERATOR_AG_ZONE(self):
        lib = MockUnit("LIBERATORAG", 50, 50, alliance=1, is_armored=True, is_flying=True)
        # Effect at (60, 50)
        eff = MockEffect(82, 60, 50)

        # --- FIX: Configure the Creep Mock with integer dimensions ---
        creep_mock = MagicMock()
        creep_mock.width = 200
        creep_mock.height = 200
        creep_mock.__getitem__ = MagicMock(return_value=0)

        obs = {
            'units': [lib], 'enemy_units': [], 'effects': [eff],
            'game_info': self.game_info, 'game_data': None,
            'creep': creep_mock  # Use the configured mock
        }

        layers = self.transformer.transform(obs)

        px_unit = int(50 / 200 * 128)
        px_zone = int(60 / 200 * 128)
        py = 128 - 1 - int(50 / 200 * 128)

        # Unit has no DPS (Liberator AG mode does not attack from unit body)
        self.assertEqual(layers[LayerConstants.LAYERS["allied_g_dps_base"], py, px_unit], 0.0)
        # Zone has DPS (Target circle)
        self.assertGreater(layers[LayerConstants.LAYERS["allied_g_dps_base"], py, px_zone], 0.0)

    def test_INVISIBLE_UNIT(self):
        """
        Test Fog of War: A cloaked enemy that is NOT detected should NOT appear
        on the feature maps. The agent shouldn't 'cheat'.
        """
        # Enemy Dark Templar, Cloaked, NOT Detected
        u = MockUnit("DARKTEMPLAR", 100, 100, alliance=4,
                     is_cloaked=True, is_detected=False)

        layers = self.run_transform(u)

        # Should be completely invisible
        self.assertEqual(self.get_val(layers, "enemy_unit_density"), 0.0,
                         "Undetected cloaked unit should be invisible")
        self.assertEqual(self.get_val(layers, "enemy_health"), 0.0)
        self.assertEqual(self.get_val(layers, "enemy_g_dps_base"), 0.0)

    def test_DAMAGED_UNIT(self):
        """
        Test Vitals: Unit with low HP should show correct normalized value.
        """
        u = MockUnit("MARINE", 100, 100, alliance=1)
        u.health_max = 45
        u.health = 10  # Nearly dead
        u.shield_max = 0
        u.shield = 0

        layers = self.run_transform(u)

        expected_hp = 10.0 / 45.0
        actual_hp = self.get_val(layers, "allied_health")

        self.assertAlmostEqual(actual_hp, expected_hp, places=4)

    def test_PSIONIC_STORM(self):
        """
        Test Effects: Psionic Storm should create a Danger Zone (DPS).
        """
        # Effect ID 1 is usually Psionic Storm in SC2
        storm_effect = MockEffect(1, 100, 100)

        layers = self.run_transform(effects=[storm_effect])

        # Storm does 28 DPS (approx) to Ground
        # The transformer treats ID 1 as Ground Threat
        val = self.get_val(layers, "enemy_g_dps_base")
        self.assertGreater(val, 0.0, "Storm should register as ground damage")

        # Should NOT appear on Allied DPS (unless we cast it, but transformer defaults effects to neutral/enemy usually)
        self.assertEqual(self.get_val(layers, "allied_g_dps_base"), 0.0)

    def test_STATIC_DEFENSE_DETECTOR(self):
        """
        Test Dual Roles: Spore Crawler is both a Weapon and a Detector.
        """
        # Enemy Spore Crawler
        u = MockUnit("SPORECRAWLER", 100, 100, alliance=4, build_progress=1.0)

        layers = self.run_transform(u)

        # 1. Check Detection Layer
        # Radius 11 is large, check px+5
        self.assertEqual(self.get_val(layers, "enemy_detection", x=self.px + 5), 1.0)

        # 2. Check DPS (Air Only)
        self.assertGreater(self.get_val(layers, "enemy_a_dps_base"), 0.0)
        self.assertEqual(self.get_val(layers, "enemy_g_dps_base"), 0.0)  # Spore doesn't hit ground

    def test_BANELING_SUICIDE(self):
        """
        Test Banelings: They have unique weapon entries (suicide).
        """
        u = MockUnit("BANELING", 100, 100, alliance=4, is_biological=True)
        layers = self.run_transform(u)

        # Baneling does massive burst damage. Transformer usually normalizes via tanh.
        # 20 base + 15 vs light.

        # Base Damage
        base_dps = self.get_val(layers, "enemy_g_dps_base")
        self.assertGreater(base_dps, 0.3, "Baneling should show base threat")

        # Bonus vs Light
        light_dps = self.get_val(layers, "enemy_g_dps_light")
        self.assertGreater(light_dps, 0.2, "Baneling should show light bonus threat")

    def test_MAP_BOUNDARIES(self):
        """
        Test Clipping: Unit at coordinate (0,0) should not crash the transformer.
        """
        u = MockUnit("MARINE", 0, 0, alliance=1)

        # This calls _get_pixel_coords.
        # (0,0) in SC2 is Bottom-Left.
        # Image (0,0) is Top-Left.
        # SC2(0,0) -> Image(0, Height) roughly.

        try:
            layers = self.run_transform(u)
            # Check bottom-left pixel of image (which corresponds to 0,0 roughly depending on axis flip)
            # Just ensure it didn't throw IndexError
            self.assertIsNotNone(layers)
        except IndexError:
            self.fail("Unit at map origin (0,0) caused IndexError")

    def test_HIGH_TEMPLAR_NO_AUTO_ATTACK(self):
        """
        Test Non-Attack Units: HT has a 'weapon' in some data patches,
        but usually it's a fake weapon or very weak.
        """
        u = MockUnit("HIGHTEMPLAR", 100, 100, alliance=4, is_psionic=True)
        layers = self.run_transform(u)

        # Check density
        self.assertEqual(self.get_val(layers, "enemy_unit_special"), 0.5)  # Psionic

        # HT has a weak auto attack in LotV, but usually negligible.
        # Let's ensure it's not registering as a massive threat like a Tank.
        dps = self.get_val(layers, "enemy_g_dps_base")
        self.assertLess(dps, 0.2, "High Templar auto attack should be weak/negligible")

if __name__ == '__main__':
    unittest.main()
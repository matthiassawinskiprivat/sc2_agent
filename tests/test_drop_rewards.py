import unittest
from unittest.mock import MagicMock
from src.level_drop_bot import LevelDropBot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class TestDropRewards(unittest.TestCase):
    def setUp(self):
        # Mock connection and config
        self.mock_conn = MagicMock()
        self.bot = LevelDropBot(self.mock_conn)
        self.bot.training_conf = {
            'rewards': {
                'load_reward': 0.1,
                'unload_reward': 1.0,
                'distance_reward_scale': 0.05
            }
        }
        self.bot.previous_cargo_sum = 0
        self.bot.previous_medivac_distances = {}
        self.bot._objective_pos = Point2((100, 100)) # Objective at (100, 100)
        self.bot.units = MagicMock()
        self.bot.enemy_structures = MagicMock()
        
    def test_progressive_load_reward(self):
        # Scenario 1: Load 1 unit
        # cargo_used: 0 -> 1
        medivac = MagicMock()
        medivac.type_id = UnitTypeId.MEDIVAC
        medivac.tag = 1
        medivac.cargo_used = 1
        medivac.position = Point2((10, 10))
        medivac.distance_to.return_value = 100.0 # Mock distance
        
        self.bot.units.filter.return_value = [medivac] # For medivac filter
        # For iteration over self.units in sum(cargo)
        self.bot.units.__iter__.return_value = [medivac] 
        
        reward = self.bot.calculate_drop_reward()
        
        # Expected: diff(1) * base(0.1) * total(1) = 0.1
        self.assertAlmostEqual(reward, 0.1)
        self.assertEqual(self.bot.previous_cargo_sum, 1)
        
        # Scenario 2: Load another unit (Total 2)
        # cargo_used: 1 -> 2
        medivac.cargo_used = 2
        reward = self.bot.calculate_drop_reward()
        
        # Expected: diff(1) * base(0.1) * total(2) = 0.2
        # Total incremental gain is higher for 2nd unit
        self.assertAlmostEqual(reward, 0.2)
        self.assertEqual(self.bot.previous_cargo_sum, 2)

    def test_cargo_weighted_distance_reward(self):
        # Scenario: Medivac with 2 units moves closer
        medivac = MagicMock()
        medivac.type_id = UnitTypeId.MEDIVAC
        medivac.tag = 1
        medivac.cargo_used = 2
        
        # Start far away
        self.bot.previous_medivac_distances[1] = 100.0
        medivac.distance_to.return_value = 90.0 # Moved 10 units closer
        
        self.bot.units.filter.return_value = [medivac]
        self.bot.units.__iter__.return_value = [medivac]
        
        # Reset cargo sum to avoid load reward interference
        self.bot.previous_cargo_sum = 2 
        
        reward = self.bot.calculate_drop_reward()
        
        # Expected: 
        # dist_diff (10) * scale (0.05) * cargo (2) = 1.0
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(self.bot.previous_medivac_distances[1], 90.0)
        
        # Scenario: Empty Medivac moves closer
        medivac.cargo_used = 0
        self.bot.previous_cargo_sum = 0
        self.bot.previous_medivac_distances[1] = 90.0
        medivac.distance_to.return_value = 80.0
        
        reward = self.bot.calculate_drop_reward()
        
        # Expected: 0.0 (No cargo = No shaping)
        self.assertAlmostEqual(reward, 0.0)

if __name__ == '__main__':
    unittest.main()

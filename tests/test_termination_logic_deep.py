import pytest
from unittest.mock import MagicMock, patch
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId

class RealMockUnit:
    def __init__(self, type_id, is_structure=False):
        self.type_id = type_id
        self.is_structure = is_structure
        self.tag = 1
        self.owner_id = 1
        self.health = 10

def test_has_combat_units_logic():
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        bot = MagicMock()
        bot.player_id = 1
        
        # Scenario 1: Only Larva
        bot.all_units = [RealMockUnit(UnitTypeId.LARVA)]
        assert env._has_combat_units(bot) is False
        
        # Scenario 2: Only Egg
        bot.all_units = [RealMockUnit(UnitTypeId.EGG)]
        assert env._has_combat_units(bot) is False
        
        # Scenario 3: Larva + Zergling
        bot.all_units = [RealMockUnit(UnitTypeId.LARVA), RealMockUnit(UnitTypeId.ZERGLING)]
        assert env._has_combat_units(bot) is True
        
        # Scenario 4: Only Pylon (structure)
        bot.all_units = [RealMockUnit(UnitTypeId.PYLON, is_structure=True)]
        assert env._has_combat_units(bot) is False
        
        # Scenario 5: Empty
        bot.all_units = []
        assert env._has_combat_units(bot) is False

        # Scenario 6: Workers and Overlords (Ignored)
        bot.all_units = [
            RealMockUnit(UnitTypeId.PROBE),
            RealMockUnit(UnitTypeId.SCV),
            RealMockUnit(UnitTypeId.DRONE),
            RealMockUnit(UnitTypeId.OVERLORD)
        ]
        assert env._has_combat_units(bot) is False

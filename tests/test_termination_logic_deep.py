import pytest
from unittest.mock import MagicMock, patch
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId

class RealMockUnit:
    def __init__(self, unit_type, is_structure=False):
        self.unit_type = unit_type
        self.is_structure = is_structure
        self.tag = 1

def test_has_combat_units_logic():
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        bot = MagicMock()
        
        # Scenario 1: Only Larva
        bot.units = [RealMockUnit(UnitTypeId.LARVA)]
        assert env._has_combat_units(bot) is False
        
        # Scenario 2: Only Egg
        bot.units = [RealMockUnit(UnitTypeId.EGG)]
        assert env._has_combat_units(bot) is False
        
        # Scenario 3: Larva + Zergling
        bot.units = [RealMockUnit(UnitTypeId.LARVA), RealMockUnit(UnitTypeId.ZERGLING)]
        assert env._has_combat_units(bot) is True
        
        # Scenario 4: Only Pylon (structure)
        bot.units = [RealMockUnit(UnitTypeId.PYLON, is_structure=True)]
        assert env._has_combat_units(bot) is False
        
        # Scenario 5: Empty
        bot.units = []
        assert env._has_combat_units(bot) is False

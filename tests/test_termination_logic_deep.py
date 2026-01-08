import pytest
from unittest.mock import MagicMock, patch
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId
from sc2.units import Units

class RealMockUnit:
    def __init__(self, type_id, is_structure=False, owner_id=1, health=10):
        self.type_id = type_id
        self.is_structure = is_structure
        self.tag = 1
        self.owner_id = owner_id
        self.health = health
        self.is_memory = False

def test_has_combat_units_logic_library():
    """
    Verify that _has_combat_units correctly filters using library-style Units object.
    """
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        bot = MagicMock()
        bot.player_id = 1
        
        # Scenario 1: Only Larva
        larva = RealMockUnit(UnitTypeId.LARVA)
        bot.all_units = Units([larva], bot)
        # We need to mock bot.units too as the implementation might use it
        bot.units = bot.all_units
        assert env._has_combat_units(bot) is False
        
        # Scenario 2: Only Egg
        egg = RealMockUnit(UnitTypeId.EGG)
        bot.all_units = Units([egg], bot)
        bot.units = bot.all_units
        assert env._has_combat_units(bot) is False
        
        # Scenario 3: Larva + Zergling
        zergling = RealMockUnit(UnitTypeId.ZERGLING)
        bot.all_units = Units([larva, zergling], bot)
        bot.units = bot.all_units
        assert env._has_combat_units(bot) is True
        
        # Scenario 4: Only Pylon (structure)
        pylon = RealMockUnit(UnitTypeId.PYLON, is_structure=True)
        bot.all_units = Units([pylon], bot)
        bot.units = bot.all_units
        assert env._has_combat_units(bot) is False
        
        # Scenario 5: Empty
        bot.all_units = Units([], bot)
        bot.units = bot.all_units
        assert env._has_combat_units(bot) is False

        # Scenario 6: Workers and Overlords (Ignored)
        probe = RealMockUnit(UnitTypeId.PROBE)
        scv = RealMockUnit(UnitTypeId.SCV)
        drone = RealMockUnit(UnitTypeId.DRONE)
        overlord = RealMockUnit(UnitTypeId.OVERLORD)
        bot.all_units = Units([probe, scv, drone, overlord], bot)
        bot.units = bot.all_units
        assert env._has_combat_units(bot) is False
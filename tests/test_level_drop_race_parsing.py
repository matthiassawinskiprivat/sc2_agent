import pytest
from sc2.data import Race
from train.envs.level_drop_env import LevelDropEnv
from unittest.mock import MagicMock

def test_parse_races_from_map_name():
    # Mocking necessary parts of LevelDropEnv to test the parsing logic
    # We bypass __init__ complexity by mocking dependencies
    with MagicMock() as mock_kwargs:
        env = LevelDropEnv.__new__(LevelDropEnv)
        
        # Test Case 1: Terran vs Terran
        agent_race, enemy_race = env._parse_races_from_map_name("01_32x32_Lower_Left_Terran_vs_Terran")
        assert agent_race == Race.Terran
        assert enemy_race == Race.Terran
        
        # Test Case 2: Protoss vs Zerg
        agent_race, enemy_race = env._parse_races_from_map_name("some_map_Protoss_vs_Zerg")
        assert agent_race == Race.Protoss
        assert enemy_race == Race.Zerg
        
        # Test Case 3: Zerg vs Protoss
        agent_race, enemy_race = env._parse_races_from_map_name("Zerg_vs_Protoss_v2")
        assert agent_race == Race.Zerg
        assert enemy_race == Race.Protoss

def test_parse_races_fallback():
    env = LevelDropEnv.__new__(LevelDropEnv)
    
    # Fallback Case: No vs delimiter
    agent_race, enemy_race = env._parse_races_from_map_name("StandardMap")
    assert agent_race == Race.Terran # Default
    assert enemy_race == Race.Terran # Default

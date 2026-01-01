import pytest
import numpy as np
from unittest.mock import MagicMock
from src.sc2_bot import SC2Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class MockUnit:
    def __init__(self, tag, pos, type_id=None):
        self.tag = tag
        self.position = pos
        self.type_id = type_id
        self.radius = 1.0
        self.is_structure = False
        self.is_enemy = True

def test_unified_observation_guidance():
    """Verify that SC2Bot uses the beacon position as command_target."""
    bot = SC2Bot(conn=MagicMock())
    
    # Mock Beacon at (20, 20)
    beacon = MockUnit(201, Point2((20, 20)), UnitTypeId.BEACON_TERRANSMALL)
    
    # Mock bot.units
    def mock_filter(type_id):
        m = MagicMock()
        if type_id == UnitTypeId.BEACON_TERRANSMALL:
            m.exists = True
            m.first = beacon
            m.__iter__.return_value = [beacon]
        else:
            m.exists = False
            m.first = None
            m.__iter__.return_value = []
        return m
    
    bot.units = MagicMock(side_effect=mock_filter)
    
    # Mock state
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    
    # Mock AgentSystem
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}
    
    # Call _get_observation
    bot._get_observation()
    
    # Check what was passed to process_observation
    call_args = bot.agent_system.process_observation.call_args
    assert call_args is not None
    raw_obs = call_args.kwargs['observation']
    
    # Verify command_target is Beacon Position
    assert raw_obs['command_target'] == beacon.position
    # Ensure it's NOT just the fallback/default None or last_target_pos
    assert raw_obs['command_target'] == Point2((20, 20))

def test_unified_observation_fallback():
    """Verify fallback if no beacon exists."""
    bot = SC2Bot(conn=MagicMock())
    
    # Set last_target_pos manually
    bot.last_target_pos = Point2((5, 5))
    
    # Mock NO Beacon
    def mock_filter(type_id):
        m = MagicMock()
        m.exists = False
        m.first = None
        m.__iter__.return_value = []
        return m
    
    bot.units = MagicMock(side_effect=mock_filter)
    bot.state = MagicMock()
    bot.state.effects = []
    bot.state.creep = None
    
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"agent_0": np.zeros((1, 64, 64))}
    
    bot._get_observation()
    
    call_args = bot.agent_system.process_observation.call_args
    raw_obs = call_args.kwargs['observation']
    
    # Should fall back to last_target_pos
    assert raw_obs['command_target'] == Point2((5, 5))

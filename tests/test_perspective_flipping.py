
import pytest
import numpy as np
from src.agent.ctf_bot import CaptureTheFlagBot
from unittest.mock import MagicMock, patch
from sc2.units import Units

def test_perspective_flipping_logic():
    bot = CaptureTheFlagBot(player_id=1)
    
    # Mock SC2 attributes
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    bot.state = MagicMock()
    
    # Create mock units
    u1 = MagicMock(tag=101, owner_id=1, is_structure=False)
    u2 = MagicMock(tag=201, owner_id=2, is_structure=False)
    
    bot.units = Units([u1], bot)
    bot.enemy_units = Units([u2], bot)
    bot.structures = Units([], bot)
    bot.enemy_structures = Units([], bot)
    bot.all_units = Units([u1, u2], bot)
    
    # Mock HierarchicalAgentSystem
    bot.agent_system = MagicMock()
    
    # Perspective 1
    bot.get_observation(perspective_player_id=1)
    args, kwargs = bot.agent_system.process_observation.call_args
    raw_obs = args[0]
    assert u1 in raw_obs['units']
    assert u2 in raw_obs['enemy_units']
    assert kwargs['player_id'] == 1
    
    # Perspective 2
    bot.get_observation(perspective_player_id=2)
    args, kwargs = bot.agent_system.process_observation.call_args
    raw_obs = args[0]
    assert u2 in raw_obs['units']
    assert u1 in raw_obs['enemy_units']
    assert kwargs['player_id'] == 2

import pytest
from unittest.mock import MagicMock, AsyncMock
from sc2.position import Point2
from sc2.ids.unit_typeid import UnitTypeId
from src.agent.ctf_bot import CaptureTheFlagBot

@pytest.fixture
def bot():
    training_conf = {
        "env": {
            "grid_size": 128
        }
    }
    bot = CaptureTheFlagBot(player_id=1, training_conf=training_conf)
    bot.game_info = MagicMock()
    bot.game_data = MagicMock()
    bot.state = MagicMock()
    bot.units = MagicMock()
    bot.enemy_units = MagicMock()
    bot.structures = MagicMock()
    bot.enemy_structures = MagicMock()
    bot.all_units = MagicMock()
    
    # Mock HierarchicalAgentSystem
    bot.agent_system = MagicMock()
    bot.agent_system.process_observation.return_value = {"active_agent": MagicMock()}
    
    # Mock 'do' to avoid strict type checking of UnitCommand
    bot.do = MagicMock()
    
    return bot

@pytest.mark.asyncio
async def test_bot_tracks_only_pylon_objective(bot):
    """Verify that only the pylon position is tracked as the command objective."""
    
    pylon_pos = Point2((50, 50))
    enemy_pylon = MagicMock()
    enemy_pylon.position = pylon_pos
    
    # Setup enemy structures to return pylon
    mock_enemy_structs = MagicMock()
    mock_enemy_structs.exists = True
    mock_enemy_structs.first = enemy_pylon
    bot.enemy_structures.return_value = mock_enemy_structs
    
    # 1. Check initial observation highlights pylon
    bot.get_observation()
    # We can't easily check the numpy array without full transformation, 
    # but we can check the raw_obs passed to agent_system
    args, _ = bot.agent_system.process_observation.call_args
    raw_obs = args[0]
    assert raw_obs['command_target'] == pylon_pos
    assert bot.enemy_pylon_pos[1] == pylon_pos

    # 2. Simulate attacking another unit at a different position
    target_pos = Point2((10, 10))
    target_unit = MagicMock()
    target_unit.position = target_pos
    bot.all_units.find_by_tag.return_value = target_unit
    
    # In CTF, apply_action is used to move/attack
    # We mock convert_spatial_action_to_world to return target_pos
    bot.agent_system.unit_agents = {"active_agent": MagicMock()}
    bot.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.return_value = (10, 10)
    
    await bot.apply_action(123, 456) # tag 123, action index 456
    
    # Verify that enemy_pylon_pos has NOT changed to target_pos
    assert bot.enemy_pylon_pos[1] == pylon_pos
    
    # 3. Simulate pylon disappearing (Fog of War)
    bot.enemy_structures.return_value.exists = False
    bot.get_observation()
    
    args, _ = bot.agent_system.process_observation.call_args
    raw_obs = args[0]
    # Should still highlight last known pylon pos
    assert raw_obs['command_target'] == pylon_pos

@pytest.mark.asyncio
async def test_bot_perspective_pylon_tracking(bot):
    """Verify that both agents see their respective enemy's pylon position."""
    # Bot 1 perspective (Trainee)
    p2_pylon_pos = Point2((80, 80))
    p2_pylon = MagicMock()
    p2_pylon.position = p2_pylon_pos
    
    # In bot 1's perspective, enemy structures is player 2's structures
    bot.enemy_structures.return_value.exists = True
    bot.enemy_structures.return_value.first = p2_pylon
    
    bot.get_observation(perspective_player_id=1)
    assert bot.enemy_pylon_pos[1] == p2_pylon_pos
    
    # Bot 2 perspective (Opponent)
    p1_pylon_pos = Point2((20, 20))
    p1_pylon = MagicMock()
    p1_pylon.position = p1_pylon_pos
    
    # In bot 2's perspective (god-mode/perspective flip), 
    # 'enemy_structures' becomes the bot's OWN structures
    bot.structures.return_value.exists = True
    bot.structures.return_value.first = p1_pylon
    
    bot.get_observation(perspective_player_id=2)
    assert bot.enemy_pylon_pos[2] == p1_pylon_pos

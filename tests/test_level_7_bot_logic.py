import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

# We will implement Level7Bot in src/level_7_bot.py
# It doesn't exist yet, so this import will fail, causing the "Red" state.
try:
    from src.level_7_bot import Level7Bot
except ImportError:
    Level7Bot = None

@pytest.mark.skipif(Level7Bot is None, reason="Level7Bot not implemented yet")
class TestLevel7Bot:
    @pytest.fixture
    def bot(self):
        # Mock configuration
        training_conf = {
            "targets": 1,
            "rewards": {
                "damage_reward": 0.1,
                "kill_reward": 10.0
            }
        }
        bot = Level7Bot(conn=MagicMock(), training_conf=training_conf)
        bot.agent_system = MagicMock()
        bot.units = MagicMock()
        bot.enemy_units = MagicMock()
        bot.client = AsyncMock()
        
        # Mock grid size for action parsing
        bot.grid_size = 10 
        # Action space = 2 * 10*10 = 200
        # 0-99: Move, 100-199: Attack
        
        return bot

    @pytest.mark.asyncio
    async def test_apply_action_move(self, bot):
        # Action 50 (Move)
        # Mock agent system conversion
        bot.agent_system.unit_agents = {"agent_0": MagicMock()}
        bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (15, 25)
        
        unit = MagicMock()
        bot.units.find_by_tag.return_value = unit
        
        # Test
        bot.do = MagicMock()
        await bot.apply_action(unit_tag=123, action=50)
        
        # Verify
        bot.units.find_by_tag.assert_called_with(123)
        unit.move.assert_called_once()
        unit.attack.assert_not_called()
        
        # Verify coordinates
        args, _ = unit.move.call_args
        target = args[0]
        assert target.x == 15 and target.y == 25

    @pytest.mark.asyncio
    async def test_apply_action_attack(self, bot):
        # Action 150 (Attack) -> 150 - 100 = 50 spatial index
        bot.agent_system.unit_agents = {"agent_0": MagicMock()}
        bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (30, 40)
        
        unit = MagicMock()
        bot.units.find_by_tag.return_value = unit
        
        bot.do = MagicMock()
        await bot.apply_action(unit_tag=123, action=150)
        
        unit.attack.assert_called_once()
        unit.move.assert_not_called()
        
        args, _ = unit.attack.call_args
        target = args[0]
        assert target.x == 30 and target.y == 40

    def test_calculate_combat_reward_damage(self, bot):
        # Setup initial state in previous_health
        enemy_tag = 999
        bot.previous_health = {enemy_tag: 50.0}
        
        # Mock current enemy state (took 10 damage)
        enemy_unit = MagicMock()
        enemy_unit.tag = enemy_tag
        enemy_unit.health = 40.0
        enemy_unit.shield = 0.0
        
        # Mock finding this unit
        bot.enemy_units = MagicMock()
        bot.enemy_units.return_value = [enemy_unit]
        
        # Calculate
        reward = bot.calculate_combat_reward()
        
        # Expected: 10 damage * 0.1 = 1.0
        assert reward == 1.0
        # Verify health updated
        assert bot.previous_health[enemy_tag] == 40.0

    def test_calculate_combat_reward_kill(self, bot):
        # Setup: Enemy exists in previous_health
        enemy_tag = 888
        bot.previous_health = {enemy_tag: 5.0}
        
        # Current state: Enemy is GONE from list (died)
        # We need to ensure logic handles "missing = dead"
        bot.enemy_units = MagicMock()
        bot.enemy_units.return_value = [] # No enemies visible
        
        reward = bot.calculate_combat_reward()
        
        # Expected: Kill reward (10.0)
        assert reward == 10.0
        # Should be removed from tracking
        assert enemy_tag not in bot.previous_health

def test_import_level_7_bot():
    # This test will definitely fail if file doesn't exist
    from src.level_7_bot import Level7Bot
    assert Level7Bot is not None
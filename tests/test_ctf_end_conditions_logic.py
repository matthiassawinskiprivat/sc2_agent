
import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from sc2.ids.unit_typeid import UnitTypeId

@pytest.fixture
def env():
    """Creates an environment instance with mocked dependencies."""
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=100)
    
    # Mock the bot
    env.bot = MagicMock()
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.client.observation = AsyncMock(return_value=MagicMock(observation=MagicMock()))
    
    # Mock bot async methods properly
    env.bot.apply_action = AsyncMock()
    
    # Mock bot methods
    env.bot.calculate_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())
    
    # Mock bot unit accessors
    env.bot.structures = MagicMock()
    env.bot.enemy_structures = MagicMock()
    env.bot.all_units = [] # Default empty
    
    # Mock Manager
    env._manager = MagicMock()
    
    # CRITICAL: Mock internal methods to bypass SC2 protocol parsing
    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value=np.zeros((23, 128, 128)))
    
    return env

@pytest.mark.asyncio
async def test_win_loss_condition_p1_lost(env):
    """Test termination when Player 1's Pylon is destroyed."""
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    env.active_player_id = 2 # Player 2's turn
    
    # P1 Pylon missing (amount == 0)
    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    
    # P2 has units, so not a draw
    env.player_units = {1: [], 2: [123]} 
    env.current_unit_index = 0
    
    # Mock alive units for P2, empty for P1?
    # Logic: p1_lost is checked first.
    # But if p1 army is also lost, does it matter?
    # Currently checks Pylon loss first.
    # But to be safe, let's mock units alive for everyone to isolate Pylon test
    def mock_units_alive():
        m = MagicMock()
        m.filter.return_value.amount = 5
        return m
    env.bot.all_units = mock_units_alive()
    
    # Execute step
    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert done is True
    assert reward == 100.0 # P2 wins on P2's turn

@pytest.mark.asyncio
async def test_continuation_on_p1_army_loss(env):
    """Test that if P1 loses army, the game continues."""
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    env.active_player_id = 1 # Player 1's turn
    
    # P1 Pylon ALIVE (ensure continuation)
    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    
    # P1 army lost
    env.player_units = {1: [], 2: [456]}
    
    # Mock units alive (P2 only)
    def mock_units_alive():
        m = MagicMock()
        # Assume filter is used to check army count
        # Logic is: filter(owner_id=1) -> amount 0
        m.filter.return_value.amount = 0 
        return m
    env.bot.all_units = mock_units_alive()
    
    # We need to ensure _has_combat_units returns False for P1
    # CaptureTheFlagSingleEnv uses self.bot.all_units.filter(...)
    # If we mock it properly:
    
    # Actually, let's just mock _has_combat_units to be sure, or rely on proper mocking of all_units
    # If we look at the implementation:
    # combat_units = self.bot.all_units.filter(...)
    # If we want to simulate P1 army loss, filter(owner_id=1) should return empty.
    
    # Let's override _has_combat_units on the instance for simplicity in this logic test
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: False if owner_id == 1 else True)

    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert done is False, "Should continue even if P1 army is lost"
    assert reward == 0.0 # No penalty
@pytest.mark.asyncio
async def test_mutual_destruction_draw(env):
    """Test termination when both sides have no units (Mutual Destruction)."""
    # New spec: Draw if units empty but pylons alive
    env.training_conf = {"rewards": {"draw_penalty": -10.0}}
    env.active_player_id = 1
    
    # Both Pylons alive
    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    
    # Both unit lists empty
    env.player_units = {1: [], 2: []}
    
    # Mock empty all_units filter for both players
    def mock_units_empty():
        m = MagicMock()
        m.filter.return_value.amount = 0
        return m
    env.bot.all_units = mock_units_empty() # Has combat units returns False
    
    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert done is True, "Should terminate on Mutual Destruction"
    assert reward == -10.0, "Should apply draw penalty"

@pytest.mark.asyncio
async def test_timeout_draw(env):
    """Test termination when max_steps is reached."""
    env.max_steps = 10
    env.steps_count = 9 # Next step will be 10
    env.training_conf = {"rewards": {"draw_penalty": -10.0}}
    
    # Normal game state
    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    env.player_units = {1: [1], 2: [2]}
    
    # Mock alive units so we don't trigger army loss
    def mock_units_alive():
        m = MagicMock()
        m.filter.return_value.amount = 5
        return m
    env.bot.all_units = mock_units_alive()
    
    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert truncated is True
    assert done is False # Truncation is not termination in Gym API usually, but logically it's game over
    # Spec says: "Timeout (Draw): Triggered when steps_count reaches max_steps."
    # And "Implement draw_penalty ... for both timeout"
    
    # Currently reward logic for timeout is missing.
    assert reward == -10.0, "Should apply draw penalty on timeout"



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
    
    # Execute step
    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert done is True
    assert reward == 100.0 # P2 wins on P2's turn

@pytest.mark.asyncio
async def test_win_loss_condition_p2_lost_on_p1_turn(env):
    """Test that if P1 loses on P1's turn, they get the loss penalty."""
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    env.active_player_id = 1 # Player 1's turn
    
    # P1 Pylon missing
    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    
    env.player_units = {1: [123], 2: [456]}
    
    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert done is True
    assert reward == -50.0 # P1 lost on their own turn
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
    
    # Currently, the code DOES NOT implement this. Expecting failure or default behavior.
    # The current implementation only checks Pylon count.
    # This test serves as the "Red" phase test for the missing feature.
    
    obs, reward, done, truncated, info = await env._async_step(0)
    
    # We WANT this to be True, but currently it's False.
    # Asserting True to confirm failure.
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
    
    obs, reward, done, truncated, info = await env._async_step(0)
    
    assert truncated is True
    assert done is False # Truncation is not termination in Gym API usually, but logically it's game over
    # Spec says: "Timeout (Draw): Triggered when steps_count reaches max_steps."
    # And "Implement draw_penalty ... for both timeout"
    
    # Currently reward logic for timeout is missing.
    assert reward == -10.0, "Should apply draw penalty on timeout"


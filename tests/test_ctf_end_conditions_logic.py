
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
    env.bot.actions = []
    env.bot.client = MagicMock()
    env.bot.client.step = AsyncMock()
    env.bot.client.actions = AsyncMock()
    env.bot.client.observation = AsyncMock(return_value=MagicMock(observation=MagicMock()))
    
    # Mock bot async methods properly
    env.bot.apply_action = AsyncMock()
    env.bot.record_pre_action_retreat_state = MagicMock()
    env.bot.do = MagicMock()
    env.bot._unit_last_order = {}
    
    # Mock bot methods
    env.bot.calculate_squad_step_reward = MagicMock(return_value=0.0)
    env.bot.get_observation = MagicMock(return_value=MagicMock())
    
    # Mock bot unit accessors
    env.bot.structures = MagicMock()
    env.bot.enemy_structures = MagicMock()
    env.bot.all_units = MagicMock()
    env.bot.all_units.find_by_tag = MagicMock(return_value=None)
    
    # Mock Manager
    env._manager = MagicMock()
    
    # CRITICAL: Mock internal methods to bypass SC2 protocol parsing
    env._refresh_bot_state = AsyncMock()
    env._refresh_unit_lists = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value=np.zeros((23, 128, 128)))
    env._maybe_promote_defender_to_attack = MagicMock()

    env.unit_pools_config = [{"agent": True, "high_level_command": "attack"}]
    env.squad_unit_tags = [(1, [100])]
    env.current_squad_index = 0
    env.current_unit_index = 0
    env.player_units = {1: [100], 2: [200]}
    
    return env


def _configure_opponent_squad(env):
    env.unit_pools_config = [{"agent": False, "high_level_command": "defend"}]
    env.squad_unit_tags = [(2, [200])]
    env.current_squad_index = 0
    env.active_player_id = 2
    env.current_unit_index = 0
    env.opponent_model = MagicMock()
    env.opponent_model.predict = MagicMock(return_value=(np.array([0, 0]), None))


def _configure_bootstrap_attack_vs_defend(env):
  """ctf_level_01 style: P1 attack marines, P2 defend + flag pylon."""
  env.unit_pools_config = [
      {"agent": True, "high_level_command": "attack", "region": "A"},
      {"high_level_command": "defend", "region": "A"},
  ]
  env.squad_unit_tags = [(1, [100]), (2, [200])]
  env.current_squad_index = 0
  env.active_player_id = 1
  env.current_unit_index = 0
  env.player_units = {1: [100], 2: [200]}


def test_pylon_loss_flags_attack_only_agent_has_no_flag(env):
    _configure_bootstrap_attack_vs_defend(env)
    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1

    p1_lost, p2_lost = env._pylon_loss_flags()

    assert p1_lost is False
    assert p2_lost is False


def test_pylon_loss_flags_defender_flag_destroyed(env):
    _configure_bootstrap_attack_vs_defend(env)
    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 0

    p1_lost, p2_lost = env._pylon_loss_flags()

    assert p1_lost is False
    assert p2_lost is True


def test_attack_only_agent_does_not_instant_lose_without_pylon(env):
    """Regression: pure attacker must not lose because it never spawned a pylon."""
    _configure_bootstrap_attack_vs_defend(env)
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    env.active_player_id = 1

    env.bot.structures(UnitTypeId.PYLON).amount = 0
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    env._has_combat_units = MagicMock(return_value=True)

    async def run():
        return await env._async_step(0)

    obs, reward, done, truncated, info = asyncio.run(run())

    assert done is False
    assert truncated is False

@pytest.mark.asyncio
async def test_win_loss_condition_p1_lost(env):
    """Test termination when a defend-pool Player 1 pylon is destroyed."""
    env.unit_pools_config = [
        {"agent": True, "high_level_command": "defend"},
        {"agent": False, "high_level_command": "attack"},
    ]
    env.squad_unit_tags = [(2, [200])]
    env.current_squad_index = 0
    env.active_player_id = 2
    env.current_unit_index = 0
    env.opponent_model = MagicMock()
    env.opponent_model.predict = MagicMock(return_value=(np.array([0, 0]), None))
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    
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
async def test_termination_on_p1_army_loss(env):
    """When the agent (P1) has no combat units left, the episode ends with loss."""
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    env.active_player_id = 1  # agent's turn

    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    env.player_units = {1: [], 2: [456]}
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: False if owner_id == 1 else True)

    obs, reward, done, truncated, info = await env._async_step(0)

    assert done is True
    assert reward == -50.0


@pytest.mark.asyncio
async def test_win_on_p2_army_loss_when_agent_acts(env):
    """When the enemy (P2) is eliminated, the agent receives win_reward."""
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}
    env.active_player_id = 1

    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    env.player_units = {1: [123], 2: []}
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: True if owner_id == 1 else False)

    obs, reward, done, truncated, info = await env._async_step(0)

    assert done is True
    assert reward == 100.0


@pytest.mark.asyncio
async def test_win_on_p1_army_loss_when_enemy_acts(env):
    """When P1 is eliminated on P2's turn, the acting opponent gets win_reward."""
    _configure_opponent_squad(env)
    env.training_conf = {"rewards": {"win_reward": 100.0, "loss_penalty": -50.0}}

    env.bot.structures(UnitTypeId.PYLON).amount = 1
    env.bot.enemy_structures(UnitTypeId.PYLON).amount = 1
    env.player_units = {1: [], 2: [456]}
    env._has_combat_units = MagicMock(side_effect=lambda owner_id: False if owner_id == 1 else True)

    obs, reward, done, truncated, info = await env._async_step(0)

    assert done is True
    assert reward == 100.0


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


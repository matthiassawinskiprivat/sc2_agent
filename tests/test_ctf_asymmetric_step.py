import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_ctf_env_asymmetric_step_loop():
    """Verify that _async_step automates non-trainee turns and returns only trainee observations."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"), \
         patch("train.envs.capture_the_flag_env.CaptureTheFlagBot") as MockBot:
        
        bot1 = MockBot.return_value
        bot1.is_trainee = True
        bot1.on_start = AsyncMock()
        bot1.player_id = 1
        
        bot2 = MagicMock()
        bot2.is_trainee = False
        bot2.on_start = AsyncMock()
        bot2.player_id = 2
        bot2.predict_action.return_value = 99
        
        MockBot.side_effect = [bot1, bot2]
        
        env = CaptureTheFlagEnv()
        env.bot1 = bot1
        env.bot2 = bot2
        
        bot1.client = MagicMock()
        bot1.client.actions = AsyncMock()
        bot1.client.step = AsyncMock()
        
        bot2.client = MagicMock()
        bot2.client.actions = AsyncMock()
        bot2.client.step = AsyncMock()
        
        env._manager = MagicMock()
        env._manager.clients = [bot1.client, bot2.client]
        
        env._resolve_map = MagicMock()
        env._synchronized_step = AsyncMock()
        env._refresh_bot_states = AsyncMock()
        env._spawn_initial_units = AsyncMock()
        
        # Setup unit lists
        # P1 has 1 unit, P2 has 1 unit
        u1 = MagicMock()
        u1.tag = 101
        u1.is_structure = False
        u2 = MagicMock()
        u2.tag = 201
        u2.is_structure = False
        
        bot1.units = [u1]
        bot2.units = [u2]
        
        # Manually refresh unit lists to populate player_units
        await env._refresh_unit_lists()
        
        # Mock bot methods
        bot1.apply_action = AsyncMock()
        bot2.apply_action = AsyncMock()
        bot1.calculate_reward.return_value = 1.0
        bot2.calculate_reward.return_value = -1.0
        bot1.get_observation.return_value = np.ones((23, 128, 128))
        bot2.get_observation.return_value = np.zeros((23, 128, 128))
        
        # Start at Player 1
        env.active_player_id = 1
        env.current_unit_index = 0
        
        # Call step
        # This should apply action to P1, then see it's P2's turn, automate P2, 
        # then return when it's P1's turn again.
        obs, reward, done, truncated, info = await env._async_step(5)
        
        # Verify P1 action applied
        bot1.apply_action.assert_called_once_with(101, 5)
        
        # Verify P2 action predicted and applied
        bot2.predict_action.assert_called_once()
        bot2.apply_action.assert_called_once_with(201, 99)
        
        # Verify physics steps (at least 2: one for P1, one for P2)
        assert env._synchronized_step.call_count >= 2
        
        # Verify we are back to Player 1
        assert env.active_player_id == 1
        assert env.current_unit_index == 0
        
        # Reward should be accumulated for P1
        assert reward >= 1.0 
        
        # Final observation should be P1's
        assert obs[0, 0, 0] == 1

@pytest.mark.asyncio
async def test_ctf_env_asymmetric_early_termination():
    """Verify that the game ends immediately if the trainee loses all units."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"), \
         patch("train.envs.capture_the_flag_env.CaptureTheFlagBot") as MockBot:
        
        bot1 = MockBot.return_value
        bot1.is_trainee = True
        bot1.player_id = 1
        bot2 = MagicMock()
        bot2.is_trainee = False
        bot2.player_id = 2
        
        MockBot.side_effect = [bot1, bot2]
        
        env = CaptureTheFlagEnv(training_conf={"rewards": {"loss_penalty": -50.0}})
        env.bot1 = bot1
        env.bot2 = bot2
        
        bot1.client = MagicMock()
        bot1.client.actions = AsyncMock()
        bot2.client = MagicMock()
        bot2.client.actions = AsyncMock()
        
        env._manager = MagicMock()
        env._synchronized_step = AsyncMock()
        env._refresh_bot_states = AsyncMock()
        
        # P1 has 0 units, P2 has 1 unit
        bot1.units = []
        bot2.units = [MagicMock()]
        await env._refresh_unit_lists()
        
        # Mock Pylons (still alive)
        p1_pylon = MagicMock()
        p2_pylon = MagicMock()
        bot1.structures.return_value = MagicMock(amount=1)
        bot2.structures.return_value = MagicMock(amount=1)
        bot1.structures.return_value.exists = True
        bot2.structures.return_value.exists = True
        
        bot1.get_observation.return_value = np.zeros((23, 128, 128))
        bot1.calculate_reward.return_value = 0.0
        
        # Call step
        obs, reward, done, truncated, info = await env._async_step(0)
        
        assert done is True
        assert reward == -50.0 # loss_penalty

@pytest.mark.asyncio
async def test_ctf_env_asymmetric_persistent_objective():
    """Verify that the game continues if ONLY the opponent loses all units."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"), \
         patch("train.envs.capture_the_flag_env.CaptureTheFlagBot") as MockBot:
        
        bot1 = MockBot.return_value
        bot1.is_trainee = True
        bot1.player_id = 1
        bot2 = MagicMock()
        bot2.is_trainee = False
        bot2.player_id = 2
        
        MockBot.side_effect = [bot1, bot2]
        
        env = CaptureTheFlagEnv()
        env.bot1 = bot1
        env.bot2 = bot2
        
        bot1.client = MagicMock()
        bot1.client.actions = AsyncMock()
        bot2.client = MagicMock()
        bot2.client.actions = AsyncMock()
        
        env._manager = MagicMock()
        env._synchronized_step = AsyncMock()
        env._refresh_bot_states = AsyncMock()
        
        # P1 has 1 unit, P2 has 0 units
        u1 = MagicMock()
        u1.tag = 101
        u1.is_structure = False
        bot1.units = [u1]
        bot2.units = []
        await env._refresh_unit_lists()
        
        # Mock Pylons (still alive)
        bot1.structures.return_value = MagicMock(amount=1)
        bot2.structures.return_value = MagicMock(amount=1)
        
        bot1.get_observation.return_value = np.zeros((23, 128, 128))
        bot1.calculate_reward.return_value = 0.0
        bot1.apply_action = AsyncMock()
        bot2.apply_action = AsyncMock()
        
        # Start at Player 1
        env.active_player_id = 1
        env.current_unit_index = 0
        
        # Call step
        obs, reward, done, truncated, info = await env._async_step(0)
        
        # Should NOT be done
        assert done is False
        assert env.active_player_id == 1 # Still P1's turn (next unit or loop back)

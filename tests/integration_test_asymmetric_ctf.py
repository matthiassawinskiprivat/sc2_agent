import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId

@pytest.mark.asyncio
async def test_integration_asymmetric_ctf_flow():
    """Verify that the env flows correctly with automated opponent in a single step call."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"), \
         patch("src.agent.ctf_bot.CaptureTheFlagBot.on_start", new_callable=AsyncMock):
        
        # We want to test the REAL logic of CaptureTheFlagEnv and CaptureTheFlagBot
        # but mock the SC2 client/connection
        env = CaptureTheFlagEnv(
            training_conf={
                "rewards": {
                    "win_reward": 100.0,
                    "loss_penalty": -50.0
                },
                "grid_size": 128
            }
        )
        
        # Setup bots manually since we aren't calling reset()
        from src.agent.ctf_bot import CaptureTheFlagBot
        env.bot1 = CaptureTheFlagBot(player_id=1, training_conf=env.training_conf)
        env.bot2 = CaptureTheFlagBot(player_id=2, training_conf=env.training_conf)
        
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Mock do method to avoid type checks and populate actions
        def mock_do(action):
            if not hasattr(env.bot1, "actions"): env.bot1.actions = []
            env.bot1.actions.append(action)
        env.bot1.do = MagicMock(side_effect=mock_do)
        
        def mock_do2(action):
            if not hasattr(env.bot2, "actions"): env.bot2.actions = []
            env.bot2.actions.append(action)
        env.bot2.do = MagicMock(side_effect=mock_do2)
        
        # Mock clients
        env.bot1.client = MagicMock()
        env.bot1.client.actions = AsyncMock()
        env.bot1.client.step = AsyncMock()
        env.bot2.client = MagicMock()
        env.bot2.client.actions = AsyncMock()
        env.bot2.client.step = AsyncMock()
        
        env._synchronized_step = AsyncMock()
        env._refresh_bot_states = AsyncMock()
        
        # Mock model for bot2
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(123), None)
        env.set_model(mock_model)
        env.bot2.model = mock_model
        
        # Mock units for bot1 and bot2
        u1 = MagicMock()
        u1.tag = 101
        u1.is_structure = False
        u1.owner_id = 1
        u1.health = 10.0
        u1.shield = 0.0
        
        u2 = MagicMock()
        u2.tag = 201
        u2.is_structure = False
        u2.owner_id = 2
        u2.health = 10.0
        u2.shield = 0.0
        
        # BurnySC2 Units object mocking is hard, let's just mock the attribute
        env.bot1.units = [u1]
        env.bot1.all_units = MagicMock()
        env.bot1.all_units.find_by_tag.return_value = u1
        
        env.bot2.units = [u2]
        env.bot2.all_units = MagicMock()
        env.bot2.all_units.find_by_tag.return_value = u2
        
        # Mock structures (Pylons)
        mock_pylon = MagicMock()
        mock_pylon.health = 10.0
        mock_pylon.shield = 10.0
        
        mock_structs = MagicMock()
        mock_structs.exists = True
        mock_structs.first = mock_pylon
        
        env.bot1.structures = MagicMock(return_value=mock_structs)
        env.bot2.structures = MagicMock(return_value=mock_structs)
        env.bot1.enemy_structures = MagicMock(return_value=mock_structs)
        env.bot2.enemy_structures = MagicMock(return_value=mock_structs)
        
        # Mock enemy units for reward calculation
        env.bot1.enemy_units = []
        env.bot2.enemy_units = []
        
        # Mock agent system for world conversion
        env.bot1.agent_system = MagicMock()
        env.bot1.agent_system.unit_agents = {"active_agent": MagicMock()}
        env.bot1.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.return_value = (10, 10)
        
        env.bot2.agent_system = MagicMock()
        env.bot2.agent_system.unit_agents = {"active_agent": MagicMock()}
        env.bot2.agent_system.unit_agents["active_agent"].convert_spatial_action_to_world.return_value = (20, 20)

        # Initial unit list refresh
        await env._refresh_unit_lists()
        
        # Start at Player 1
        env.active_player_id = 1
        env.current_unit_index = 0
        
        # Call step
        # 1. P1 moves unit 101.
        # 2. Env advances to P2 unit 201.
        # 3. Env automates P2 unit 201.
        # 4. Env advances back to P1 unit 101.
        obs, reward, done, truncated, info = await env._async_step(5)
        
        assert env.active_player_id == 1
        assert env.current_unit_index == 0
        assert mock_model.predict.call_count == 1
        
        # Verify that both actions were applied to their respective units
        # P1's unit
        env.bot1.client.actions.assert_called()
        # P2's unit
        env.bot2.client.actions.assert_called()

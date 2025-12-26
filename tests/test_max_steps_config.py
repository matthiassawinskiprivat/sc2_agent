
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from train.envs.level_0_env import Level0Env

class MockSC2Process:
    def __init__(self, **kwargs):
        pass
    async def __aenter__(self):
        controller = AsyncMock()
        controller.create_game = AsyncMock()
        return controller
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def _close_connection(self):
        pass
    def _clean(self):
        pass

class MockClient:
    def __init__(self, ws):
        pass
    async def ping(self):
        pass
    async def join_game(self, **kwargs):
        pass
    async def get_game_info(self):
        game_info = MagicMock()
        game_info._proto = MagicMock()
        return game_info
    async def get_game_data(self):
        return MagicMock()
    async def observation(self):
        obs = MagicMock()
        obs.observation = MagicMock()
        return obs
    async def step(self):
        pass
    async def actions(self, actions):
        pass

class TestMaxStepsConfig:
    def test_level0_env_accepts_max_steps(self):
        """
        Test that Level0Env accepts max_steps argument and sets it correctly.
        """
        max_steps = 100
        
        # This is expected to fail currently because Level0Env.__init__ 
        # does not accept max_steps
        try:
            env = Level0Env(
                level_name="level_0",
                visualize=False,
                grid_size=32,
                sc2_process_cls=MockSC2Process,
                client_cls=MockClient,
                max_steps=max_steps
            )
            assert env.max_steps == max_steps
        except TypeError as e:
            pytest.fail(f"Level0Env does not accept max_steps argument: {e}")

    def test_level0_env_truncation(self):
        """
        Test that the environment actually truncates at max_steps.
        We force max_steps manually here to test the logic separate from the init bug,
        or we assume the init bug is fixed. 
        Actually, for TDD, let's try to set it via init.
        """
        max_steps = 10
        
        try:
            env = Level0Env(
                level_name="level_0",
                visualize=False,
                grid_size=32,
                sc2_process_cls=MockSC2Process,
                client_cls=MockClient,
                max_steps=max_steps
            )
        except TypeError:
            # If init fails, we manually set it to test the logic part
            # But the goal of THIS test is to ensure the parameter is respected.
            # So failing at init is a valid failure for this test.
            pytest.fail("Level0Env does not accept max_steps argument")

        # Mock the async loop parts to avoid real game launch
        env._launch_sc2_process = AsyncMock()
        env._setup_game_session = AsyncMock()
        env._reset_bot_state = AsyncMock()
        
        # Mock bot
        env._bot = MagicMock()
        env._bot._get_observation.return_value = (None, {})
        env._bot._get_step_results.return_value = (None, 0, False, False, {})
        env._bot.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world.return_value = (0, 0)
        
        env._loop = asyncio.new_event_loop()
        
        env.reset()
        
        # Step until max_steps - 1
        for _ in range(max_steps - 1):
            _, _, done, truncated, _ = env.step(0)
            assert not done
            assert not truncated
            
        # The next step should trigger truncation
        _, _, done, truncated, _ = env.step(0)
        assert truncated
        assert done

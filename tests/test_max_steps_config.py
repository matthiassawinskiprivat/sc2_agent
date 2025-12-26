
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
        
        env = Level0Env(
            level_name="level_0",
            visualize=False,
            grid_size=32,
            sc2_process_cls=MockSC2Process,
            client_cls=MockClient,
            max_steps=max_steps
        )
        assert env.max_steps == max_steps

    def test_level0_env_truncation(self):
        """
        Test that the environment actually truncates at max_steps.
        """
        max_steps = 10
        
        env = Level0Env(
            level_name="level_0",
            visualize=False,
            grid_size=32,
            sc2_process_cls=MockSC2Process,
            client_cls=MockClient,
            max_steps=max_steps
        )

        # Mock the async loop parts to avoid real game launch
        env._launch_sc2_process = AsyncMock()
        env._setup_game_session = AsyncMock()
        env._reset_bot_state = AsyncMock()
        
        # Mock _async_step directly to return dummy results
        env._async_step = AsyncMock(return_value=(None, 0, False, False, {}))
        
        # Mock bot for reset
        env._bot = MagicMock()
        env._bot._get_observation.return_value = (None, {})
        
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
        assert env.steps == max_steps

    def test_level0_env_default_max_steps(self):
        """
        Test that Level0Env falls back to default max_steps (300).
        """
        env = Level0Env(
            level_name="level_0",
            visualize=False,
            grid_size=32,
            sc2_process_cls=MockSC2Process,
            client_cls=MockClient
        )
        assert env.max_steps == 300

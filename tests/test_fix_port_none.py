import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from train.envs.level_0_env import Level0Env

class MockSC2ProcessNone:
    """Simulates a process that fails to get a port once, then succeeds."""
    attempt_count = 0
    def __init__(self, *args, **kwargs):
        MockSC2ProcessNone.attempt_count += 1
        self.port = None if MockSC2ProcessNone.attempt_count == 1 else 12345
        self.ws = "ws://127.0.0.1:12345/sc2api" if self.port else None
        
    async def __aenter__(self):
        return MagicMock()
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.mark.asyncio
async def test_launch_retry_on_none_port():
    """
    Verifies that _launch_sc2_process retries if port is None.
    """
    env = Level0Env(
        sc2_process_cls=MockSC2ProcessNone,
        client_cls=MagicMock(),
        level_name="level_0"
    )
    
    # We need to mock _discover_maps to avoid FileNotFoundError
    env._discover_maps = MagicMock(return_value=["MapName"])
    env.map_name = "MapName"
    
    # Reset attempt count
    MockSC2ProcessNone.attempt_count = 0
    
    await env._launch_sc2_process()
    
    # Should have attempted twice (first failed with None, second succeeded)
    assert MockSC2ProcessNone.attempt_count == 2
    assert env._sc2_process.port == 12345
    print("\nSUCCESS: SC2 process launch retried and succeeded after port=None")

if __name__ == "__main__":
    asyncio.run(test_launch_retry_on_none_port())

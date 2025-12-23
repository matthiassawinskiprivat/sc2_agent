
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
import asyncio
from sc2.protocol import ConnectionAlreadyClosed
from aiohttp.client_exceptions import WSMessageTypeError

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.envs import level_0_env

class TestEnvRecovery:
    @patch('train.envs.level_0_env.Level0Env._launch_sc2_process', new_callable=AsyncMock)
    @patch('train.envs.level_0_env.Level0Env._setup_game_session', new_callable=AsyncMock)
    @patch('asyncio.new_event_loop')
    def test_reset_retries_on_connection_error(self, mock_new_loop, mock_setup, mock_launch):
        """
        Verify that reset() retries if setup fails with ConnectionAlreadyClosed or WSMessageTypeError.
        """
        env = level_0_env.Level0Env(
            level_name="level_0",
            visualize=False,
            grid_size=32,
            num_workers=1,
            worker_index=0
        )
        
        # Mock bot to avoid AttributeError later
        env._bot = MagicMock()
        env._bot._get_observation.return_value = (None, None)
        
        # Mock loop
        mock_loop_instance = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_new_loop.return_value = mock_loop_instance
        
        # Create an exception to simulate
        failure_exc = ConnectionAlreadyClosed("Simulated failure")
        
        # We need to simulate the flow:
        # 1. reset() -> launch() -> setup() (FAILS)
        # 2. catch -> close() -> reset() (RECURSIVE or LOOP)
        # 3. reset() -> launch() -> setup() (SUCCEEDS)
        
        # We can simulate this by having run_until_complete raise the first time _setup_game_session's coro is passed,
        # and succeed the second time.
        
        # But run_until_complete receives a coroutine object. It's hard to tell which one it is just by looking at the arg.
        # However, we can track calls.
        
        # Let's verify that _setup_game_session is called TWICE.
        # And that the exception is NOT propagated out of reset().
        
        # We need run_until_complete to raise ONLY when the failing setup coro is passed.
        
        # To make this easier, let's just make run_until_complete raise based on a counter, 
        # mimicking the underlying coroutine failing.
        
        # Call 1: launch (success)
        # Call 2: setup (FAIL)
        # Call 3: close (cleanup) - implicitly called via self.close()
        # Call 4: launch (success)
        # Call 5: setup (success)
        
        failures = [None, failure_exc, None, None, None] # launch, setup(fail), close?, launch, setup
        
        # Wait, run_until_complete is called for launch and setup.
        # self.close() calls run_until_complete for _sc2_process.__aexit__ too.
        
        call_counter = 0
        
        def run_side_effect(coro):
            nonlocal call_counter
            idx = call_counter
            call_counter += 1
            
            # Simple heuristic: if it's the "fail" index, raise
            # 0: launch
            # 1: setup -> FAIL
            # 2: close (maybe?)
            # 3: launch
            # 4: setup -> SUCCESS
            
            if idx == 1:
                raise failure_exc
            
            # Restore bot on any subsequent call (retry)
            if idx > 1:
                env._bot = MagicMock()
                env._bot._get_observation.return_value = (None, None)

            return None

        mock_loop_instance.run_until_complete.side_effect = run_side_effect
        
        # We expect NO exception to be raised, because it should handle it and retry.
        # Currently, it raises, so this test should FAIL.
        env.reset()
        
        # Verify it retried
        assert call_counter > 2
        assert mock_setup.call_count >= 1 # It might be called even if run_until_complete raises before executing it? 
        # Actually run_until_complete raises, simulating the execution failure.
        
        # If it retried, we expect at least 3 calls to run_until_complete (launch, setup-fail, launch-retry...)
        
if __name__ == "__main__":
    pytest.main([__file__])

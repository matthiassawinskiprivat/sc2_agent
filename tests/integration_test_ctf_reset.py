
import pytest
import asyncio
import os
import sys
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from src.utils.sc2_cleanup import kill_switch_sc2

def test_ctf_real_reset():
    """
    Perform a real reset of the CaptureTheFlagEnv.
    This requires SC2 to be installed and available in the path.
    """
    # Ensure clean state
    kill_switch_sc2()
    
    # We use a reduced max_steps and grid_size for faster testing
    env = CaptureTheFlagEnv(
        map_name="BotMicroArena_6",
        headless=False,
        realtime=False
    )
    
    try:
        logger.info("Attempting CTF Environment Reset...")
        # reset() in MultiClientEnv is synchronous but calls internal async methods
        obs, info = env.reset()
        
        assert obs is not None
        logger.info(f"Reset successful! Obs shape: {obs.shape}")
        
        # Take a few steps to ensure communication remains open
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            assert obs is not None
            logger.info(f"Step {i+1} successful.")
            
    except Exception as e:
        logger.error(f"CTF Environment Reset/Step failed: {e}")
        raise e
    finally:
        env.close()
        kill_switch_sc2()

if __name__ == "__main__":
    test_ctf_real_reset()

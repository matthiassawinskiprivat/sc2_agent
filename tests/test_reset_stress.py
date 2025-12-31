
import pytest
import time
import os
import sys
from loguru import logger

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.envs import level_0_env

# Configure logger to capture output if run as script
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

class TestResetStress:
    def test_rapid_resets(self):
        """
        Stress test to reproduce WSMessageTypeError by performing rapid resets.
        This simulates the map rotation logic in the training pipeline.
        """
        # Force hard reset on every episode
        # level_0_env.HARD_RESET_COUNT = 1
        
        num_resets = 20 # Reduced count since hard resets take longer
        env = None
        
        try:
            logger.info("Starting rapid reset stress test...")
            # Initialize environment with small grid size to speed up setup
            env = level_0_env.Level0Env(
                level_name="level_0",
                visualize=False,
                grid_size=32,
                num_workers=1,
                worker_index=0
            )

            for i in range(num_resets):
                logger.info(f"--- Reset #{i+1}/{num_resets} ---")
                obs, info = env.reset()
                assert obs is not None
                
                # Perform a few steps to ensure game is actually running
                for _ in range(5):
                    action = env.action_space.sample()
                    env.step(action)
                
                # We don't explicit close here, mimicking standard RL loop reset
                # But Level0Env has logic to hard reset periodically. 
                # To force stress, we might want to manually trigger hard resets 
                # or rely on the environment's internal counter if it's small enough.
                
                # Let's check the internal counter in Level0Env. 
                # It hard resets every HARD_RESET_COUNT (default 10).
                # We can manually set the counter to force hard resets more often if needed,
                # or just run enough iterations. 50 iterations with count 10 = 5 hard resets.
                
        except Exception as e:
            logger.exception(f"Stress test failed at reset #{i+1}")
            pytest.fail(f"Stress test failed: {e}")
        finally:
            if env:
                env.close()
                logger.info("Environment closed.")

if __name__ == "__main__":
    pytest.main([__file__])

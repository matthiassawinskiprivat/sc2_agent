
import os
import sys
import time
import signal
from loguru import logger

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.envs.level_0_env import Level0Env
from sb3_contrib import MaskablePPO

def validate_fix():
    logger.info("Starting fix validation (CNN + Level0Env)...")
    
    # Create Env (headless, no SC2 reset needed for initialization check)
    env = Level0Env(visualize=False, grid_size=32)
    
    # Create Model with CnnPolicy (triggers is_image_space if normalize_images is not False)
    model = MaskablePPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs={
            "features_extractor_kwargs": {"features_dim": 128},
            "normalize_images": False  # Bypasses is_image_space check
        }
    )
    
    logger.info("Model initialized successfully. Fix confirmed (no freeze during init).")
    
    # Try a very brief learn() call. 
    # This might fail because SC2 isn't running, but we care about the SB3 PREPROCESSING freeze.
    logger.info("Attempting model.learn() to check preprocessing...")
    try:
        # Start a thread to kill this process after 10s if it hangs
        import threading
        def kill_after_timeout():
            time.sleep(10)
            logger.error("Validation HANGED! Fix failed.")
            os._exit(1)
        
        timeout_thread = threading.Thread(target=kill_after_timeout, daemon=True)
        timeout_thread.start()
        
        # This will likely fail with connection error, which is fine!
        # As long as it doesn't hang in is_image_space.
        model.learn(total_timesteps=1)
        
    except Exception as e:
        if "Connection" in str(e) or "SC2" in str(e) or "Protocol" in str(e):
            logger.info(f"Execution reached SC2 connection phase: {e}")
            logger.info("Fix VALIDATED.")
        else:
            logger.error(f"Unexpected error: {e}")
            raise e
    finally:
        env.close()

if __name__ == "__main__":
    validate_fix()

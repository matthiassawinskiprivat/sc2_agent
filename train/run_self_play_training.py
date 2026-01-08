
import os
import sys
import glob
from stable_baselines3 import PPO
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.envs.capture_the_flag_self_play_env import CaptureTheFlagSelfPlayEnv

class SelfPlayCallback:
    """
    Custom callback to update the opponent model periodically.
    """
    def __init__(self, env, update_interval=10000, save_dir="models/self_play"):
        self.env = env
        self.update_interval = update_interval
        self.last_update_step = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, locals_, globals_):
        num_timesteps = locals_['self'].num_timesteps
        
        if num_timesteps - self.last_update_step >= self.update_interval:
            logger.info(f"Self-Play: Updating Opponent Model at step {num_timesteps}...")
            
            # Save current model
            model_path = os.path.join(self.save_dir, f"model_{num_timesteps}.zip")
            locals_['self'].save(model_path)
            
            # Load as new opponent (frozen)
            # We load it into a separate PPO instance to keep it independent
            opponent_model = PPO.load(model_path)
            
            # Update env
            # Handle VecEnv wrapping if necessary
            # Assuming DummyVecEnv or SubprocVecEnv, we need to call method on underlying env
            # But for simplicity, assuming single env or handling it:
            env = self.env
            if hasattr(env, "venv"):
                env = env.venv.envs[0] # Unwrap
            elif hasattr(env, "envs"):
                env = env.envs[0]
            
            # If wrapped in Monitor/etc, unwrap
            while hasattr(env, "env"):
                if isinstance(env, CaptureTheFlagSelfPlayEnv):
                    break
                env = env.env
                
            if isinstance(env, CaptureTheFlagSelfPlayEnv):
                env.set_opponent_model(opponent_model)
                logger.info("Opponent model updated successfully.")
            else:
                logger.warning("Could not find CaptureTheFlagSelfPlayEnv to update opponent.")

            self.last_update_step = num_timesteps
        return True

def main():
    logger.info("Starting Self-Play Training Loop...")
    
    # 1. Create Env
    # Using the self-play env
    env = CaptureTheFlagSelfPlayEnv(
        map_name="08_64x64_Full", 
        level_name="capture_the_flag_0",
        visualize=False,
        realtime=False
    )
    
    # 2. Setup Model
    # Load latest or create new
    model_dir = "models/checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    
    # Find latest model to resume or start fresh
    checkpoints = sorted(glob.glob(f"{model_dir}/*.zip"), key=os.path.getmtime)
    if checkpoints:
        latest_model = checkpoints[-1]
        logger.info(f"Loading latest model: {latest_model}")
        model = PPO.load(latest_model, env=env, verbose=1)
        
        # Initialize Opponent with the SAME model
        opponent_model = PPO.load(latest_model)
        env.set_opponent_model(opponent_model)
    else:
        logger.info("Starting fresh training...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/tensorboard")
        
        # Initialize Opponent with a fresh copy (or random)
        # We can save the fresh model and load it
        model.save("models/temp_init")
        opponent_model = PPO.load("models/temp_init")
        env.set_opponent_model(opponent_model)

    # 3. Train
    callback = SelfPlayCallback(env, update_interval=5000) # Update opponent every 5000 steps
    
    try:
        model.learn(total_timesteps=100000, callback=callback)
        model.save("models/final_self_play")
        logger.info("Training finished.")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()

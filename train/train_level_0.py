import sys
import os
import glob
from loguru import logger
from stable_baselines3.common.monitor import Monitor

# Clean logs
logger.remove()
logger.add(sys.stderr, level="INFO")

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
from train.envs.level_0_env import Level0Env

# Directories
MODEL_DIR = "models/level_0"
LOG_DIR = "logs/level_0"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    env = Level0Env(visualize=False)
    return Monitor(env)


def get_latest_model():
    final_path = os.path.join(MODEL_DIR, "final_model.zip")
    if os.path.exists(final_path): return final_path
    checkpoints = glob.glob(os.path.join(MODEL_DIR, "zerg_*_steps.zip"))
    if not checkpoints: return None

    def get_steps(filename):
        try:
            return int(filename.split("_")[-2])
        except:
            return 0

    return max(checkpoints, key=get_steps)


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule function.
    """

    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and decreases to 0.0 during training
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Reduced to 4 to save RAM (Buffer size depends on this)
    NUM_ENVS = 4
    # Reduced steps per update to lower RAM usage
    N_STEPS = 256
    BATCH_SIZE = 512  # n_steps * num_envs / 2

    print(f"Launching {NUM_ENVS} parallel SC2 processes...", flush=True)
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)], start_method='spawn')

    # 2. Model Setup
    latest_model_path = get_latest_model()

    print("Initializing PPO Model...", flush=True)

    if latest_model_path:
        print(f"\n!!! RESUMING TRAINING FROM: {latest_model_path} !!!\n", flush=True)
        model = MaskablePPO.load(
            latest_model_path,
            env=env,
            device="cuda",
            tensorboard_log=LOG_DIR,
            custom_objects={
                "learning_rate": 3e-4,
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE
            }
        )
    else:
        print("\n!!! STARTING NEW TRAINING SESSION !!!\n", flush=True)
        model = MaskablePPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=linear_schedule(1e-4),
            batch_size=BATCH_SIZE,
            n_steps=N_STEPS,
            gamma=0.99,
            policy_kwargs={
                "features_extractor_kwargs": {"features_dim": 512}
            },
            device="cuda",
            ent_coef=0.01
        )

    print("Model Initialized. Starting Learn Loop...", flush=True)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODEL_DIR, name_prefix="zerg")

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=(latest_model_path is None)
    )

    model.save(f"{MODEL_DIR}/final_model")
    env.close()
    print("Training finished.")

import sys
import os
import glob
import yaml
from loguru import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from sb3_contrib import MaskablePPO
import torch

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The pipeline will now use the original, stable Level0Env
from train.envs.level_0_env import Level0Env
from train.callbacks import StopTrainingOnMetricThreshold
from train.subproc_vec_env_logger import SubprocVecEnvLogger


# Directories
CONFIG_DIR = "config"
MODEL_DIR_BASE = "models"
LOG_DIR_BASE = "logs"

# Clean logs
logger.remove()
logger.add(sys.stderr, level="INFO")

def get_latest_model(model_dir):
    """Gets the path to the latest model in a directory."""
    final_path = os.path.join(model_dir, "final_model.zip")
    if os.path.exists(final_path):
        logger.info(f"Found final model: {final_path}")
        return final_path
    
    checkpoints = glob.glob(os.path.join(model_dir, "zerg_*_steps.zip"))
    if not checkpoints:
        return None

    return max(checkpoints, key=os.path.getmtime)

def train_pipeline():
    """Main training pipeline."""
    with open(os.path.join(CONFIG_DIR, "pipeline.yaml"), 'r') as f:
        pipeline_config = yaml.safe_load(f)

    logger.info(f"Starting training pipeline for levels: {pipeline_config['levels']}")
    
    last_model_path = None
    last_grid_size = None

    # Get all available levels from the config directory
    all_level_files = sorted(glob.glob(os.path.join(CONFIG_DIR, "level_*.yaml")))
    all_levels = [os.path.splitext(os.path.basename(f))[0] for f in all_level_files]


    for level_name in pipeline_config["levels"]:
        logger.info(f"--- Starting Level: {level_name} ---")

        # 1. Load Level Configuration
        with open(os.path.join(CONFIG_DIR, f"{level_name}.yaml"), 'r') as f:
            level_config = yaml.safe_load(f)

        training_params = level_config["training"]
        env_params = level_config["env"]
        model_params = level_config["model"]

        # Check for grid size change
        current_grid_size = env_params["grid_size"]
        if last_grid_size is not None and current_grid_size != last_grid_size:
            logger.warning(f"Grid size changed from {last_grid_size} to {current_grid_size}. "
                           f"The model from the previous level will be loaded, but with a new policy head.")
        last_grid_size = current_grid_size

        model_dir = os.path.join(MODEL_DIR_BASE, level_name)
        log_dir = os.path.join(LOG_DIR_BASE, level_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Find model from previous level
        previous_level_model_path = None
        current_level_index = all_levels.index(level_name)
        if current_level_index > 0:
            previous_level_name = all_levels[current_level_index-1]
            previous_level_model_dir = os.path.join(MODEL_DIR_BASE, previous_level_name)
            previous_level_model_path = get_latest_model(previous_level_model_dir)

        # 2. Create Vectorized Environment
        num_envs = training_params["num_envs"]
        headless = env_params.get("headless", True)

        vec_env_cls = SubprocVecEnv
        vec_env_kwargs = {'start_method': 'spawn'}
        
        if not headless:
            logger.warning("Running in non-headless mode. Forcing num_envs=1 and using DummyVecEnv.")
            num_envs = 1
            vec_env_cls = DummyVecEnv
            vec_env_kwargs = {}
        
        logger.info(f"Creating {num_envs} environment(s) using {vec_env_cls.__name__}...")

        env_fns = [
            lambda i=i: Monitor(
                Level0Env(
                    level_name=level_name,
                    visualize=not headless,
                    grid_size=env_params["grid_size"],
                    worker_index=i,
                    num_workers=num_envs
                )
            )
            for i in range(num_envs)
        ]
        
        env = vec_env_cls(env_fns, **vec_env_kwargs)

        try:
            # 3. Setup or Load Model
            # If there's a model from a previous level, use it (curriculum learning)
            model_path_to_load = last_model_path or previous_level_model_path or get_latest_model(model_dir)

            if model_path_to_load:
                logger.info(f"Resuming/Loading training from: {model_path_to_load}")
                
                # Check if action space is different
                saved_model_policy = MaskablePPO.load(model_path_to_load).policy
                logger.info(f"Current env action space: {env.action_space.n}")
                logger.info(f"Saved model action space: {saved_model_policy.action_space.n}")
                if env.action_space.n != saved_model_policy.action_space.n:
                    logger.warning("Action space has changed. Transferring weights to a new model.")
                    
                    # Load the old model
                    old_model = MaskablePPO.load(model_path_to_load)
                    
                    # Create a new model with the new environment
                    model = MaskablePPO(
                        policy=model_params["policy"],
                        env=env,
                        verbose=1,
                        tensorboard_log=log_dir,
                        learning_rate=training_params["learning_rate"],
                        batch_size=training_params["batch_size"],
                        n_steps=training_params["n_steps"],
                        gamma=model_params["gamma"],
                        ent_coef=model_params["ent_coef"],
                        n_epochs=model_params["n_epochs"],
                        policy_kwargs=model_params.get("policy_kwargs", {}),
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    
                    # Copy the weights from the old model's feature extractor to the new model's feature extractor
                    model.policy.features_extractor.load_state_dict(old_model.policy.features_extractor.state_dict())
                    
                else:
                    custom_objects = {
                        "learning_rate": training_params["learning_rate"],
                        "n_steps": training_params["n_steps"],
                        "batch_size": training_params["batch_size"],
                    }
                    model = MaskablePPO.load(
                        model_path_to_load,
                        env=env,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        tensorboard_log=log_dir,
                        custom_objects=custom_objects
                    )
            else:
                logger.info("Starting new training session with MaskablePPO.")
                model = MaskablePPO(
                    policy=model_params["policy"],
                    env=env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    learning_rate=training_params["learning_rate"],
                    batch_size=training_params["batch_size"],
                    n_steps=training_params["n_steps"],
                    gamma=model_params["gamma"],
                    ent_coef=model_params["ent_coef"],
                    n_epochs=model_params["n_epochs"],
                    policy_kwargs=model_params.get("policy_kwargs", {}),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

            # 4. Train Model
            logger.info(f"Training for up to {training_params['training_steps']} timesteps...")
            
            checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=model_dir, name_prefix="zerg")
            
            callback_list = [checkpoint_callback]

            if vec_env_cls == SubprocVecEnv:
                callback_list.append(SubprocVecEnvLogger())

            if "evaluation" in level_config and "stop_on_metric" in level_config["evaluation"]:
                eval_params = level_config["evaluation"]["stop_on_metric"]
                logger.info(f"Adding stopping condition: Stop when '{eval_params['monitor']}' is {eval_params['comparison']} {eval_params['threshold']} for {eval_params['patience']} consecutive checks.")
                stop_callback = StopTrainingOnMetricThreshold(
                    monitor=eval_params['monitor'],
                    threshold=eval_params['threshold'],
                    comparison=eval_params['comparison'],
                    patience=eval_params['patience'],
                    verbose=1
                )
                callback_list.append(stop_callback)

            model.learn(
                total_timesteps=training_params["training_steps"],
                callback=CallbackList(callback_list),
                reset_num_timesteps=not bool(model_path_to_load)
            )

            # 5. Save Final Model
            logger.info("Training for this level complete. Saving final model.")
            final_model_path = os.path.join(model_dir, "final_model.zip")
            model.save(final_model_path)
            last_model_path = final_model_path # For the next level in the curriculum

        finally:
            # 6. Clean Up
            logger.info("Closing environments for this level.")
            env.close()
        logger.info(f"--- Finished Level: {level_name} ---")

    logger.info("Training pipeline finished.")

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is not installed. Please install it with 'pip install pyyaml'")
        sys.exit(1)
        
    train_pipeline()

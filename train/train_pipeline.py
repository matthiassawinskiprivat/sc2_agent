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

class PipelineManager:
    """
    Orchestrates the multi-level training pipeline.
    The manager checks each level defined in pipeline.yaml and determines the course of action.
    - If a 'final_model.zip' exists for a level, it's considered complete and skipped.
    - If no final model exists but checkpoints (e.g., 'zerg_*_steps.zip') are found,
      training will resume from the latest checkpoint.
    - If no models are found for a level, a new training session is initiated.
      For curriculum learning, it will use the 'final_model.zip' from the last successfully
      completed level as a starting point.
    - The pipeline finishes when all levels are completed.
    """

    def __init__(self, config_dir=CONFIG_DIR, model_dir_base=MODEL_DIR_BASE, log_dir_base=LOG_DIR_BASE):
        self.config_dir = config_dir
        self.model_dir_base = model_dir_base
        self.log_dir_base = log_dir_base
        
        with open(os.path.join(self.config_dir, "pipeline.yaml"), 'r') as f:
            self.pipeline_config = yaml.safe_load(f)
            
        self.all_level_files = sorted(glob.glob(os.path.join(self.config_dir, "level_*.yaml")))
        self.all_levels = [os.path.splitext(os.path.basename(f))[0] for f in self.all_level_files]

    def get_level_status(self, level_name):
        """
        Determines the training status of a given level.

        Returns:
            A tuple (status, path):
            - status (str): "COMPLETED", "CONTINUE_TRAINING", or "NEEDS_TRAINING".
            - path (str or None): Path to the relevant model file if it exists.
        """
        model_dir = os.path.join(self.model_dir_base, level_name)
        final_model_path = os.path.join(model_dir, "final_model.zip")

        if os.path.exists(final_model_path):
            logger.info(f"Level '{level_name}' is already completed. Found final model: {final_model_path}")
            return "COMPLETED", final_model_path

        checkpoints = glob.glob(os.path.join(model_dir, "zerg_*_steps.zip"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Found checkpoint for level '{level_name}'. Resuming training from: {latest_checkpoint}")
            return "CONTINUE_TRAINING", latest_checkpoint

        logger.info(f"No models found for level '{level_name}'. Needs new training session.")
        return "NEEDS_TRAINING", None

    def _train_level(self, level_name, model_path_to_load=None):
        """
        Trains a single level.
        This method contains the core logic for setting up the environment,
        loading/creating a model, training it, and saving the final result.
        """
        logger.info(f"--- Starting Training for Level: {level_name} ---")
        
        # 1. Load Level Configuration
        with open(os.path.join(self.config_dir, f"{level_name}.yaml"), 'r') as f:
            level_config = yaml.safe_load(f)

        training_params = level_config["training"]
        env_params = level_config["env"]
        model_params = level_config["model"]

        model_dir = os.path.join(self.model_dir_base, level_name)
        log_dir = os.path.join(self.log_dir_base, level_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

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
        
        logger.info(f"Creating {num_envs} environment(s) for level '{level_name}' using {vec_env_cls.__name__}...")

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
            if model_path_to_load:
                logger.info(f"Resuming/Loading training from: {model_path_to_load}")
                
                # Check if action space is different
                saved_model_policy = MaskablePPO.load(model_path_to_load).policy
                logger.info(f"Current env action space: {env.action_space.n}")
                logger.info(f"Saved model action space: {saved_model_policy.action_space.n}")
                if env.action_space.n != saved_model_policy.action_space.n:
                    logger.warning("Action space has changed. Transferring weights to a new model.")
                    
                    old_model = MaskablePPO.load(model_path_to_load)
                    
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
            return final_model_path

        finally:
            # 6. Clean Up
            logger.info("Closing environments for this level.")
            env.close()
        logger.info(f"--- Finished Level: {level_name} ---")


    def run(self):
        """Executes the entire training pipeline."""
        logger.info(f"Starting training pipeline for levels: {self.pipeline_config['levels']}")
        last_final_model_path = None

        # Find the last completed level to start curriculum from, even across sessions
        for i in range(len(self.pipeline_config['levels']) -1, -1, -1):
            level = self.pipeline_config['levels'][i]
            status, path = self.get_level_status(level)
            if status == "COMPLETED":
                last_final_model_path = path
                break


        for level_name in self.pipeline_config["levels"]:
            logger.info(f"--- Processing Level: {level_name} ---")
            status, model_path = self.get_level_status(level_name)

            if status == "COMPLETED":
                last_final_model_path = model_path
                continue
            
            # Determine the model to load for curriculum learning
            model_to_load = None
            if status == "CONTINUE_TRAINING":
                model_to_load = model_path # Continue from the checkpoint
            elif status == "NEEDS_TRAINING":
                # If it's a new training, try to load the model from the previous level
                model_to_load = last_final_model_path
            
            # Train the level
            final_model_path = self._train_level(level_name, model_path_to_load=model_to_load)
            last_final_model_path = final_model_path

        logger.info("Training pipeline finished.")


def train_pipeline():
    """Main training pipeline."""
    manager = PipelineManager()
    manager.run()

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is not installed. Please install it with 'pip install pyyaml'")
        sys.exit(1)
        
    train_pipeline()

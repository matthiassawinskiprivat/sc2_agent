import sys
import os
import glob
import yaml
import shutil
import subprocess
from typing import List, Any
from loguru import logger
import torch
import numpy as np
from gymnasium import spaces

# Disable distribution validation
torch.distributions.Distribution.set_default_validate_args(False)
torch.set_num_threads(8)

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_pipeline import PipelineManager, ENV_REGISTRY, BOT_REGISTRY, MaskableSubprocVecEnv, MaskableSyncVecEnv
from train.env_utils import EnvFactory
from src.sc2_bot_universal import SC2BotUniversal

# Register the specialized bot globally for this process
BOT_REGISTRY["SC2Bot"] = SC2BotUniversal
from src.agent.maskable_ppo import MaskablePPO
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.models.universal_policy import UniversalActorCriticPolicy
from train.callbacks import StopTrainingOnMetricThreshold, DiagnosticsCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.utils.watchdog import Watchdog
from src.utils.sc2_cleanup import kill_switch_sc2

class UniversalPipelineManager(PipelineManager):
    def __init__(self, config_dir="config_universal", model_dir_base="models_universal", log_dir_base="logs_universal"):
        # Explicitly ensure bases are relative to project root if needed
        # But since we run from root via .bat, simple names are fine.
        super().__init__(config_dir=config_dir, model_dir_base=model_dir_base, log_dir_base=log_dir_base)
        self.agent_architecture = "universal"
        logger.info(f"UniversalPipelineManager initialized (Targeting {config_dir})")

    def _train_level(self, level_name, model_path_to_load=None, reset_timesteps=True):
        logger.info(f"--- [UNIVERSAL] Starting Training for Level: {level_name} ---")
        
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
        
        if "policy_kwargs" in model_params:
            pk = model_params["policy_kwargs"]
            if "features_extractor_class" in pk and pk["features_extractor_class"] == "src.agent.models.unit_control_network.UnitControlNetwork":
                pk["features_extractor_class"] = UnitControlNetwork

        env_class_name = env_params.get("env_class", "Level0Env")
        EnvClass = ENV_REGISTRY.get(env_class_name)
        
        if not headless:
            num_envs = 1
            vec_env_cls = MaskableSyncVecEnv
        else:
            vec_env_cls = MaskableSubprocVecEnv if num_envs > 1 else MaskableSyncVecEnv

        combined_conf = training_params.copy()
        combined_conf.update(env_params)
        combined_conf["is_training"] = True
        combined_conf["agent_architecture"] = "universal"
        combined_conf["config_dir"] = self.config_dir 
        combined_conf["spatial_resolution"] = env_params.get("spatial_resolution", 16)

        env_fns = [
            EnvFactory(
                EnvClass, level_name, not headless, env_params["grid_size"], i, num_envs,
                self.sc2_process_cls, self.client_cls, combined_conf, env_params.get("max_steps", 300)
            )
            for i in range(num_envs)
        ]
        
        env = vec_env_cls(env_fns)

        try:
            # 3. Setup Model
            algo_name = model_params.get("algorithm", "MaskablePPO")
            AlgoClass = MaskableRecurrentPPO if algo_name == "MaskableRecurrentPPO" else MaskablePPO
            
            # Universal-specific Policy kwargs
            policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
            policy_kwargs["spatial_resolution"] = combined_conf["spatial_resolution"]
            
            model_kwargs = {
                "policy": UniversalActorCriticPolicy,
                "env": env,
                "verbose": 1,
                "tensorboard_log": log_dir,
                "learning_rate": training_params["learning_rate"],
                "batch_size": training_params["batch_size"],
                "n_steps": training_params["n_steps"],
                "gamma": model_params["gamma"],
                "ent_coef": model_params["ent_coef"],
                "n_epochs": model_params["n_epochs"],
                "policy_kwargs": policy_kwargs,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "level_name": level_name,
            }
            if AlgoClass == MaskableRecurrentPPO:
                model_kwargs["rnn_type"] = model_params.get("rnn_type", "lstm")
            
            if model_path_to_load:
                logger.info(f"Loading weights from: {model_path_to_load}")
                model = AlgoClass.load(
                    model_path_to_load, env=env, device=model_kwargs["device"],
                    tensorboard_log=log_dir,
                    custom_objects={"learning_rate": training_params["learning_rate"]}
                )
            else:
                model = AlgoClass(**model_kwargs)

            # 4. Train
            total_steps_target = training_params["training_steps"]
            watchdog = Watchdog(timeout=training_params.get("watchdog_timeout", 3600.0), recurring=True)
            watchdog.start()

            # Calculate save frequency to be exactly 50,000 global steps
            save_freq_steps = 50000
            save_freq_calls = max(1, save_freq_steps // num_envs)

            checkpoint_callback = CheckpointCallback(save_freq=save_freq_calls, save_path=model_dir, name_prefix="universal")
            diagnostics_callback = DiagnosticsCallback(check_freq=1024, watchdog=watchdog)
            
            model.learn(
                total_timesteps=total_steps_target,
                callback=CallbackList([checkpoint_callback, diagnostics_callback]),
                reset_num_timesteps=reset_timesteps
            )

            final_model_path = os.path.join(model_dir, "final_model.zip")
            model.save(final_model_path)
            return final_model_path

        finally:
            if 'watchdog' in locals(): watchdog.stop()
            env.close()

    def run(self):
        """Executes the entire universal training pipeline."""
        logger.info(f"Starting universal training pipeline for levels: {self.pipeline_config['levels']}")
        kill_switch_sc2()
        
        last_final_model_path = None
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
            
            model_to_load = status == "CONTINUE_TRAINING" and model_path or last_final_model_path
            reset_timesteps = status != "CONTINUE_TRAINING"
            if status == "NEEDS_TRAINING": self._cleanup_logs(level_name)
            
            final_model_path = self._train_level(level_name, model_path_to_load=model_to_load, reset_timesteps=reset_timesteps)
            last_final_model_path = final_model_path

        logger.info("Universal Training pipeline finished.")
        kill_switch_sc2()

def main():
    manager = UniversalPipelineManager()
    manager.run()

if __name__ == "__main__":
    main()

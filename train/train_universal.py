import sys
import os
import copy
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

from train.train_pipeline import (
    PipelineManager,
    ENV_REGISTRY,
    BOT_REGISTRY,
    MaskableSubprocVecEnv,
    MaskableSyncVecEnv,
    cleanup_worker_logs,
)
from train.env_utils import EnvFactory
from src.sc2_bot_universal import SC2BotUniversal

# Register the specialized bot globally for this process
BOT_REGISTRY["SC2Bot"] = SC2BotUniversal
from src.agent.maskable_ppo import MaskablePPO
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.models.universal_policy import UniversalActorCriticPolicy
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.utils.rollout_buffer_factory import sync_rollout_buffer_config
from train.callbacks import StopTrainingOnMetricThreshold, DiagnosticsCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.utils.watchdog import Watchdog
from src.utils.sc2_cleanup import kill_switch_sc2
from src.utils.training_device import resolve_training_device, warmup_cuda_if_needed

class UniversalPipelineManager(PipelineManager):
    def __init__(self, config_dir="config_universal", model_dir_base="models_universal", log_dir_base="logs_universal"):
        # Explicitly ensure bases are relative to project root if needed
        # But since we run from root via .bat, simple names are fine.
        super().__init__(config_dir=config_dir, model_dir_base=model_dir_base, log_dir_base=log_dir_base)
        self.agent_architecture = "universal"
        logger.info(f"UniversalPipelineManager initialized (Targeting {config_dir})")

    def _train_level(self, level_name, model_path_to_load=None, reset_timesteps=True, merged_config=None):
        logger.info(f"--- [UNIVERSAL] Starting Training for Level: {level_name} ---")
        
        # 1. Load Level Configuration (use pipeline-merged config when provided — cumulative layers, etc.)
        if merged_config is not None:
            level_config = copy.deepcopy(merged_config)
        else:
            with open(os.path.join(self.config_dir, f"{level_name}.yaml"), 'r') as f:
                level_config = yaml.safe_load(f)

        training_params = level_config["training"]
        env_params = level_config["env"]
        model_params = level_config["model"]

        model_dir = os.path.join(self.model_dir_base, level_name)
        tb_sub = training_params.get("tensorboard_subdir", "")
        log_dir = (
            os.path.join(self.log_dir_base, level_name, tb_sub)
            if tb_sub
            else os.path.join(self.log_dir_base, level_name)
        )
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
        combined_conf["spatial_resolution"] = 32

        # Combined initial steps for curriculum (extract from checkpoint name if possible)
        initial_steps = 0
        if model_path_to_load:
            filename = os.path.basename(model_path_to_load)
            if "steps" in filename:
                try:
                    parts = filename.split("_")
                    for p in parts:
                        if p.isdigit():
                            initial_steps = int(p)
                            break
                    logger.info(f"Resuming level '{level_name}' from global step count: {initial_steps}")
                except Exception as e:
                    logger.warning(f"Could not parse step count from checkpoint filename '{filename}': {e}")

        # Universal spatial policy is 32×32; env.grid_size is legacy (bot/env bookkeeping only).
        _grid = env_params.get("grid_size", 32)
        env_fns = [
            EnvFactory(
                EnvClass, level_name, not headless, _grid, i, num_envs,
                self.sc2_process_cls, self.client_cls, combined_conf, env_params.get("max_steps", 300),
                initial_steps=initial_steps
            )
            for i in range(num_envs)
        ]
        
        logger.info(
            f"Creating {num_envs} env(s) with {vec_env_cls.__name__} for '{level_name}' "
            f"(Subproc join timeout {os.environ.get('SB3_SUBPROC_JOIN_TIMEOUT', '120')}s)..."
        )
        if vec_env_cls is MaskableSyncVecEnv:
            env = vec_env_cls(env_fns)
        else:
            env = vec_env_cls(env_fns, start_method="spawn")
        logger.info(f"Vectorized environment ready for '{level_name}'.")

        try:
            train_device = resolve_training_device(training_params)
            warmup_cuda_if_needed(train_device)
            logger.info(f"Sync universal training device: {train_device}")

            # 3. Setup Model
            algo_name = model_params.get("algorithm", "MaskablePPO")
            AlgoClass = MaskableRecurrentPPO if algo_name == "MaskableRecurrentPPO" else MaskablePPO
            PolicyClass = UniversalRecurrentPolicy if AlgoClass == MaskableRecurrentPPO else UniversalActorCriticPolicy
            
            # Universal-specific Policy kwargs
            policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
            policy_kwargs["spatial_resolution"] = combined_conf["spatial_resolution"]
            
            rollout_buffer_class, rollout_buffer_kwargs = sync_rollout_buffer_config(
                training_params, env_params
            )
            model_kwargs = {
                "policy": PolicyClass,
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
                "device": train_device,
                "level_name": level_name,
                "rollout_buffer_class": rollout_buffer_class,
                "rollout_buffer_kwargs": rollout_buffer_kwargs,
            }
            if AlgoClass == MaskableRecurrentPPO:
                model_kwargs["rnn_type"] = model_params.get("rnn_type", "lstm")
            model_kwargs["target_kl"] = training_params.get(
                "target_kl", model_params.get("target_kl")
            )

            if model_path_to_load:
                logger.info(f"Loading weights from: {model_path_to_load}")
                try:
                    load_kwargs = {
                        "env": env,
                        "device": model_kwargs["device"],
                        "tensorboard_log": log_dir,
                        "custom_objects": {
                            "learning_rate": training_params["learning_rate"]
                        },
                        "rollout_buffer_class": rollout_buffer_class,
                        "rollout_buffer_kwargs": rollout_buffer_kwargs,
                    }
                    if AlgoClass == MaskableRecurrentPPO:
                        load_kwargs["rnn_type"] = model_kwargs.get("rnn_type", "lstm")
                    # Same-level resume (CONTINUE_TRAINING) only; curriculum uses fresh Adam.
                    load_kwargs["restore_optimizer"] = not reset_timesteps
                    model = AlgoClass.load(model_path_to_load, **load_kwargs)
                except Exception as e:
                    logger.warning(
                        f"Could not load weights from '{model_path_to_load}' "
                        f"(likely architecture change between levels): {e}. Starting fresh."
                    )
                    model = AlgoClass(**model_kwargs)
            else:
                model = AlgoClass(**model_kwargs)

            from src.utils.model_utils import maybe_apply_loud_command_head_init

            maybe_apply_loud_command_head_init(
                model,
                level_config,
                resume=bool(model_path_to_load and not reset_timesteps),
            )

            model.learner_profile = training_params.get("learner_profile", True)

            # 4. Train
            total_steps_target = training_params["training_steps"]
            steps_to_train = total_steps_target
            
            if model_path_to_load and not reset_timesteps:
                steps_already_done = model.num_timesteps
                steps_to_train = total_steps_target - steps_already_done
                logger.info(f"Resuming training. Target: {total_steps_target}, Already done: {steps_already_done}, Remaining: {steps_to_train}")
                
                if steps_to_train <= 0:
                    logger.info("Training already reached target steps. Saving and finishing.")
                    final_model_path = os.path.join(model_dir, "final_model.zip")
                    model.save(final_model_path)
                    return final_model_path

            watchdog = Watchdog(timeout=training_params.get("watchdog_timeout", 3600.0), recurring=True)
            watchdog.start()

            # Calculate save frequency to be exactly 50,000 global steps
            save_freq_steps = 50000
            save_freq_calls = max(1, save_freq_steps // num_envs)

            checkpoint_callback = CheckpointCallback(save_freq=save_freq_calls, save_path=model_dir, name_prefix="universal")
            diagnostics_callback = DiagnosticsCallback(check_freq=1024, watchdog=watchdog)
            
            model.learn(
                total_timesteps=steps_to_train,
                callback=CallbackList([checkpoint_callback, diagnostics_callback]),
                reset_num_timesteps=reset_timesteps
            )

            final_model_path = os.path.join(model_dir, "final_model.zip")
            model.save(final_model_path)
            return final_model_path

        finally:
            if 'watchdog' in locals():
                watchdog.stop()
            logger.info(
                f"[{level_name}] Closing vectorized env (drain timeout "
                f"{os.environ.get('SB3_SUBPROC_RECV_DRAIN_TIMEOUT', '30')}s per worker)..."
            )
            env.close()
            logger.info(f"[{level_name}] Vectorized env closed.")

    def run(self):
        """Executes the entire universal training pipeline with additive configuration."""
        logger.info(f"Starting universal training pipeline for levels: {self.pipeline_config['levels']}")
        kill_switch_sc2()
        cleanup_worker_logs()
        
        last_final_model_path = None
        # Initialize base_config with pipeline-level settings
        self.cumulative_config = {
            "agent_architecture": "universal",
            "active_layers": [],
            "active_tactical_layers": [],
            "action_heads": [],
            "categorical_heads": [],
        }

        # Find the last completed level to start curriculum from
        for i in range(len(self.pipeline_config['levels']) -1, -1, -1):
            level = self.pipeline_config['levels'][i]
            status, path = self.get_level_status(level)
            if status == "COMPLETED":
                last_final_model_path = path
                break

        for level_name in self.pipeline_config["levels"]:
            logger.info(f"--- Processing Level: {level_name} ---")
            
            # Load level config and update cumulative state
            with open(os.path.join(self.config_dir, f"{level_name}.yaml"), 'r') as f:
                level_config = yaml.safe_load(f)
            
            # ADDITIVE LOGIC: Merge level config into cumulative config
            # This ensures subsequent levels 'inherit' layers from previous ones
            env_cfg = level_config.get("env", {})

            if "active_layers" in env_cfg:
                for layer in env_cfg["active_layers"]:
                    if layer not in self.cumulative_config["active_layers"]:
                        self.cumulative_config["active_layers"].append(layer)

            if "active_tactical_layers" in env_cfg:
                for layer in env_cfg["active_tactical_layers"]:
                    if layer not in self.cumulative_config["active_tactical_layers"]:
                        self.cumulative_config["active_tactical_layers"].append(layer)

            if "action_heads" in env_cfg:
                for head in env_cfg["action_heads"]:
                    if head not in self.cumulative_config["action_heads"]:
                        self.cumulative_config["action_heads"].append(head)

            if "categorical_heads" in env_cfg:
                for head in env_cfg["categorical_heads"]:
                    if head not in self.cumulative_config["categorical_heads"]:
                        self.cumulative_config["categorical_heads"].append(head)

            # Store the current state for this training session
            level_config["active_layers"] = self.cumulative_config["active_layers"].copy()
            level_config["active_tactical_layers"] = self.cumulative_config["active_tactical_layers"].copy()
            level_config["action_heads"] = self.cumulative_config["action_heads"].copy()
            level_config["categorical_heads"] = self.cumulative_config["categorical_heads"].copy()
            
            status, model_path = self.get_level_status(level_name)
            if status == "COMPLETED":
                last_final_model_path = model_path
                continue
            
            model_to_load = status == "CONTINUE_TRAINING" and model_path or last_final_model_path
            reset_timesteps = status != "CONTINUE_TRAINING"
            if status == "NEEDS_TRAINING":
                self._cleanup_logs(level_name)
            elif status == "CONTINUE_TRAINING" and model_path:
                from train.training_log_utils import truncate_logs_for_checkpoint_resume

                tb_sub = level_config.get("training", {}).get("tensorboard_subdir", "")
                log_dir = (
                    os.path.join(self.log_dir_base, level_name, tb_sub)
                    if tb_sub
                    else os.path.join(self.log_dir_base, level_name)
                )
                try:
                    truncate_logs_for_checkpoint_resume(log_dir, model_path)
                except Exception as exc:
                    logger.warning(
                        f"Could not truncate resume logs in {log_dir}: {exc}"
                    )
            
            # Pass the merged config to train_level
            final_model_path = self._train_level(level_name, model_path_to_load=model_to_load, reset_timesteps=reset_timesteps, merged_config=level_config)
            last_final_model_path = final_model_path
            # Orphan SC2 instances (common with SC2_SKIP_KILL_SWITCH=1) hold ports and cause the next
            # level's SubprocVecEnv workers to hang in launch. Force kill between levels unless opted out.
            if os.environ.get("SC2_SKIP_KILL_BETWEEN_LEVELS", "0") != "1":
                logger.info(f"SC2 cleanup after level '{level_name}' (before next curriculum stage)...")
                kill_switch_sc2(force=True)

        logger.info("Universal Training pipeline finished.")
        kill_switch_sc2()

def main():
    manager = UniversalPipelineManager()
    manager.run()

if __name__ == "__main__":
    main()

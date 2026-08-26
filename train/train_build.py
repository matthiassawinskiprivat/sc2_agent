"""Synchronous build-agent training pipeline."""

import copy
import os
import sys

import torch
import yaml
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_pipeline import PipelineManager, MaskableSubprocVecEnv, MaskableSyncVecEnv
from train.env_utils import EnvFactory
from train.envs.build_env import BuildEnv
from train.envs.zerg_build_env import ZergBuildEnv
from src.agent.maskable_ppo import MaskablePPO
from src.agent.models.build_policy import BuildActorCriticPolicy, BuildFeaturesExtractor
from src.utils.training_device import resolve_training_device, warmup_cuda_if_needed
from src.utils.rollout_buffer_factory import sync_rollout_buffer_config
from src.utils.model_utils import load_build_checkpoint
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

torch.distributions.Distribution.set_default_validate_args(False)

BUILD_ENV_REGISTRY = {
    "BuildEnv": BuildEnv,
    "ZergBuildEnv": ZergBuildEnv,
}


class BuildPipelineManager(PipelineManager):
    # Never wipe machine-wide SC2 — build must coexist with universal/control training.
    global_sc2_kill_on_boundaries = False

    def __init__(
        self,
        config_dir: str = "config_build_zerg",
        model_dir_base: str = "models_build_zerg",
        log_dir_base: str = "logs_build_zerg",
    ):
        super().__init__(
            config_dir=config_dir,
            model_dir_base=model_dir_base,
            log_dir_base=log_dir_base,
        )

    def _train_level(
        self,
        level_name,
        model_path_to_load=None,
        reset_timesteps=True,
        merged_config=None,
    ):
        logger.info(f"--- [BUILD] Starting Training for Level: {level_name} ---")

        if merged_config is not None:
            level_config = copy.deepcopy(merged_config)
        else:
            with open(os.path.join(self.config_dir, f"{level_name}.yaml"), "r") as f:
                level_config = yaml.safe_load(f)

        training_params = level_config["training"]
        env_params = level_config["env"]
        model_params = level_config["model"]

        pipeline_training = self.pipeline_config.get("training", {})
        if isinstance(pipeline_training, dict) and "num_envs" in pipeline_training:
            training_params = {**training_params, "num_envs": pipeline_training["num_envs"]}

        model_dir = os.path.join(self.model_dir_base, level_name)
        log_dir = os.path.join(self.log_dir_base, level_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Match universal pipeline: on CONTINUE_TRAINING, drop TB/CSV metrics past
        # the resume checkpoint so curves do not double-stack after restart.
        if model_path_to_load and not reset_timesteps:
            try:
                from train.async_train_universal import free_port
                from train.training_log_utils import truncate_logs_for_checkpoint_resume

                tb_port = int(os.environ.get("SC2_TENSORBOARD_PORT", "6006"))
                free_port(tb_port)
                truncate_logs_for_checkpoint_resume(log_dir, model_path_to_load)
            except Exception as exc:
                logger.warning(
                    f"Could not truncate resume logs in {log_dir} "
                    f"(checkpoint={model_path_to_load!r}): {exc}"
                )

        num_envs = training_params["num_envs"]
        headless = env_params.get("headless", True)
        env_class_name = env_params.get("env_class", "ZergBuildEnv")
        # Alternating curriculum: manager levels use the manager trainer.
        if "Manager" in str(env_class_name):
            from train.train_manager import ManagerPipelineManager

            mgr = ManagerPipelineManager(
                config_dir=self.config_dir,
                model_dir_base=self.model_dir_base,
                log_dir_base=self.log_dir_base,
            )
            mgr.pipeline_config = self.pipeline_config
            mgr.sc2_process_cls = self.sc2_process_cls
            mgr.client_cls = self.client_cls
            return mgr._train_level(
                level_name,
                model_path_to_load=None,  # fresh manager Adam
                reset_timesteps=True,
                merged_config=level_config,
            )

        EnvClass = BUILD_ENV_REGISTRY.get(env_class_name)
        if EnvClass is None:
            raise ValueError(f"Unknown build env class: {env_class_name}")

        logger.info(
            f"[BUILD] Level '{level_name}': num_envs={num_envs}, "
            f"training_steps={training_params['training_steps']}"
        )

        if "policy_kwargs" in model_params:
            pk = model_params["policy_kwargs"]
            fe = pk.get("features_extractor_class")
            if isinstance(fe, str) and "BuildFeaturesExtractor" in fe:
                pk["features_extractor_class"] = BuildFeaturesExtractor

        vec_env_cls = (
            MaskableSyncVecEnv
            if (not headless or num_envs == 1)
            else MaskableSubprocVecEnv
        )

        combined_conf = training_params.copy()
        combined_conf.update(env_params)
        combined_conf["is_training"] = True
        combined_conf["agent_architecture"] = "build"
        combined_conf["config_dir"] = self.config_dir
        combined_conf["race"] = combined_conf.get("race", "zerg")

        initial_steps = 0
        if model_path_to_load and "steps" in os.path.basename(model_path_to_load):
            for part in os.path.basename(model_path_to_load).split("_"):
                if part.isdigit():
                    initial_steps = int(part)
                    break

        env_fns = [
            EnvFactory(
                EnvClass,
                level_name,
                not headless,
                16,
                i,
                num_envs,
                self.sc2_process_cls,
                self.client_cls,
                combined_conf,
                env_params.get("max_steps", 300),
                initial_steps=initial_steps,
            )
            for i in range(num_envs)
        ]
        env = vec_env_cls(env_fns) if vec_env_cls is MaskableSyncVecEnv else vec_env_cls(
            env_fns, start_method="spawn"
        )

        train_device = resolve_training_device(training_params)
        warmup_cuda_if_needed(train_device)

        policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
        rollout_buffer_class, rollout_buffer_kwargs = sync_rollout_buffer_config(
            training_params, env_params, namespace="build"
        )
        rollout_buffer_kwargs["enable_rnn_sampling"] = False
        model_kwargs = {
            "policy": BuildActorCriticPolicy,
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

        if model_path_to_load:
            model = load_build_checkpoint(
                model_path_to_load,
                model_kwargs,
                load_kwargs={
                    "env": env,
                    "device": train_device,
                    "tensorboard_log": log_dir,
                    "rollout_buffer_class": rollout_buffer_class,
                    "rollout_buffer_kwargs": rollout_buffer_kwargs,
                },
            )
        else:
            model = MaskablePPO(**model_kwargs)

        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(training_params["n_steps"], 10_000),
                    save_path=model_dir,
                    name_prefix="build",
                )
            ]
        )
        model.learn(
            total_timesteps=training_params["training_steps"],
            callback=callbacks,
            reset_num_timesteps=reset_timesteps,
        )
        final_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_path)
        env.close()
        return final_path


if __name__ == "__main__":
    BuildPipelineManager().run()

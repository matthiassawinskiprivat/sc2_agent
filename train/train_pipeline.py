import sys
import os
import glob
import yaml
import shutil
import subprocess
from typing import List, Any
from loguru import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.agent.maskable_ppo import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
import torch

# Disable distribution validation to avoid Simplex errors with MaskablePPO
# caused by numerical precision issues when masking logits with -1e8
torch.distributions.Distribution.set_default_validate_args(False)

# Optimization: Limit PyTorch CPU threads to reduce orchestration overhead
# and context switching during the training phase.
torch.set_num_threads(8)

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The pipeline will now use the original, stable Level0Env
from train.envs.level_0_env import Level0Env
from train.envs.level_3_env import Level3Env
from train.envs.level_7_env import Level7Env
from train.envs.level_drop_env import LevelDropEnv
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv as CaptureTheFlagEnv
from train.callbacks import StopTrainingOnMetricThreshold, DiagnosticsCallback
from train.sync_vec_env import SyncVecEnv
from src.utils.watchdog import Watchdog
from src.utils.sc2_cleanup import kill_switch_sc2
from src.utils.model_utils import graft_model_weights
from train.env_utils import EnvFactory, get_action_masks

# Import Custom Network
from src.agent.models.unit_control_network import UnitControlNetwork

# Specialized Bots for Environment initialization
from src.level_0_bot import Level0Bot
from src.level_3_bot import Level3Bot
from src.level_7_bot import Level7Bot
from src.level_drop_bot import LevelDropBot
from src.agent.ctf_bot import CaptureTheFlagBot

# Environment Registry
ENV_REGISTRY = {
    "Level0Env": Level0Env,
    "Level3Env": Level3Env,
    "Level7Env": Level7Env, # Note: This looks like a bug in original (mapped to Bot instead of Env)
    "LevelDropEnv": LevelDropEnv,
    "CaptureTheFlagEnv": CaptureTheFlagEnv
}

# Bot Registry for environments that instantiate their own bot (Level0, Level3, Level7)
BOT_REGISTRY = {
    "Level0Env": Level0Bot,
    "Level3Env": Level3Bot,
    "Level7Env": Level7Bot,
    "LevelDropEnv": LevelDropBot,
    "CaptureTheFlagEnv": CaptureTheFlagBot
}

# Bot Registry for environments that instantiate their own bot (Level0, Level3, Level7)
BOT_REGISTRY = {
    "Level0Env": Level0Bot,
    "Level3Env": Level3Bot,
    "Level7Env": Level7Bot,
    "CaptureTheFlagEnv": CaptureTheFlagBot
}

# Directories
CONFIG_DIR = "config"
MODEL_DIR_BASE = "models"
LOG_DIR_BASE = "logs"
WORKER_LOG_DIR = "logs"


def cleanup_worker_logs(logs_dir: str = WORKER_LOG_DIR) -> int:
    """
    Delete accumulated ``worker_*.log`` files from prior training runs.

    Env workers write to ``logs/worker_<id>.log`` with a fresh random id each
    process, so these pile up across starts/resumes. Call once at session start
    (new training or resume) so only the current run's worker logs remain.
    """
    pattern = os.path.join(logs_dir, "worker_*.log")
    deleted = 0
    for path in glob.glob(pattern):
        try:
            os.remove(path)
            deleted += 1
        except OSError as e:
            logger.warning(f"Could not delete worker log {path}: {e}")
    if deleted:
        logger.info(f"Deleted {deleted} worker log file(s) from {logs_dir}/")
    return deleted

# Clean logs
if __name__ == "__main__":
    logger.remove()
    log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    logger.add(sys.stderr, level=log_level, colorize=True)

class MaskableSubprocVecEnv(SyncVecEnv):
    """
    SubprocVecEnv doesn't natively support get_action_masks. 
    We wrap it to add support, inheriting from SyncVecEnv to share the get_action_masks logic 
    if possible, but actually SubprocVecEnv is quite different. 
    Actually, let's just implement it directly or inherit from SubprocVecEnv.
    """
    pass # Wait, let's re-evaluate. 

# Re-defining more cleanly:
from stable_baselines3.common.vec_env import SubprocVecEnv
from train.sb3_subproc_worker_safe import apply_sb3_subproc_worker_patch
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
import numpy as np

apply_sb3_subproc_worker_patch()

class MaskableSubprocVecEnv(SubprocVecEnv):
    def get_action_masks(self) -> np.ndarray:
        logger.debug("MaskableSubprocVecEnv: get_action_masks() called.")
        # Explicitly cast to bool to ensure compatibility with MaskablePPO
        masks = np.stack(self.env_method("action_masks")).astype(bool)
        logger.debug("MaskableSubprocVecEnv: get_action_masks() completed.")
        return masks

    def get_attr(self, attr_name: str, indices = None) -> List[Any]:
        """
        Return attribute from vectorized environment.
        Intersects 'action_masks' to avoid pickling bound methods from workers.
        """
        if attr_name == "action_masks":
            # SB3's is_masking_supported() only checks if the attribute exists.
            # We return a dummy picklable value directly from the main process.
            # The actual logic remains in the action_masks() method called via env_method.
            target_remotes = self._get_target_remotes(indices)
            return [True for _ in target_remotes]
            
        return super().get_attr(attr_name, indices)

    def step_wait(self) -> VecEnvStepReturn:
        logger.debug("MaskableSubprocVecEnv: step_wait() waiting for workers...")
        results = super().step_wait()
        logger.debug("MaskableSubprocVecEnv: step_wait() received results.")
        return results

    def reset(self) -> np.ndarray:
        logger.debug("MaskableSubprocVecEnv: reset() waiting for workers...")
        try:
            results = super().reset()
            logger.debug("MaskableSubprocVecEnv: reset() completed.")
            return results
        except (EOFError, BrokenPipeError) as e:
            logger.error("MaskableSubprocVecEnv: Worker process died during reset().")
            # We don't re-raise here if we want the higher level to handle it, 
            # but usually it's better to re-raise with a better message.
            raise e

    def close(self) -> None:
        """
        Shutdown worker processes without leaving zombie SC2 workers or broken pipes.
        SB3's default close() joins forever; if a worker is stuck in env.reset() after
        a terminal step or in env.close(), the next level can start while old workers
        still hit BrokenPipeError on send. We drain/ignore pipe errors, then join with
        a timeout and terminate stuck processes (important on Windows).
        """
        if self.closed:
            return
        join_timeout = float(os.environ.get("SB3_SUBPROC_JOIN_TIMEOUT", "120"))
        recv_drain_s = float(os.environ.get("SB3_SUBPROC_RECV_DRAIN_TIMEOUT", "30"))
        try:
            if self.waiting:
                # Never block forever: SB3 can leave waiting=True if step_wait never ran (e.g. crash),
                # or a worker can be stuck in env.reset() inside the step handler before sending.
                for idx, remote in enumerate(self.remotes):
                    try:
                        if hasattr(remote, "poll") and not remote.poll(recv_drain_s):
                            logger.warning(
                                f"MaskableSubprocVecEnv: no step reply from worker {idx} within {recv_drain_s}s; "
                                "skipping recv (worker likely stuck in SC2 reset/close)."
                            )
                            continue
                        remote.recv()
                    except (EOFError, BrokenPipeError, OSError):
                        pass
                self.waiting = False
            for remote in self.remotes:
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for process in self.processes:
                process.join(timeout=join_timeout)
                if process.is_alive():
                    logger.warning(
                        f"MaskableSubprocVecEnv: worker pid={process.pid} did not exit after {join_timeout}s; terminating."
                    )
                    process.terminate()
                    process.join(timeout=30)
                    if process.is_alive():
                        try:
                            process.kill()
                        except AttributeError:
                            pass
                        process.join(timeout=10)
        finally:
            self.closed = True

class MaskableSyncVecEnv(SyncVecEnv):
    def get_action_masks(self) -> np.ndarray:
        return np.stack(self.env_method("action_masks")).astype(bool)

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

    # When True, run() calls kill_switch_sc2() at start/end (wipes all SC2 on the machine).
    # BuildPipelineManager sets False so build training can run beside universal/control agents.
    global_sc2_kill_on_boundaries = True

    def __init__(self, config_dir=CONFIG_DIR, model_dir_base=MODEL_DIR_BASE, log_dir_base=LOG_DIR_BASE, 
                 sc2_process_cls=None, client_cls=None):
        self.config_dir = config_dir
        self.model_dir_base = model_dir_base
        self.log_dir_base = log_dir_base
        
        # Injected dependencies for testing
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        
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

        checkpoints = glob.glob(os.path.join(model_dir, "*_steps.zip"))
        if checkpoints:
            from src.utils.model_utils import latest_step_checkpoint

            latest_checkpoint = latest_step_checkpoint(checkpoints)
            logger.info(f"Found checkpoint for level '{level_name}'. Resuming training from: {latest_checkpoint}")
            return "CONTINUE_TRAINING", latest_checkpoint

        logger.info(f"No models found for level '{level_name}'. Needs new training session.")
        return "NEEDS_TRAINING", None

    def _cleanup_logs(self, start_level_name):
        """
        Deletes TensorBoard logs for the given level and all subsequent levels
        defined in the pipeline. This ensures a clean TensorBoard view when
        re-training from a specific point.
        """
        levels = self.pipeline_config.get("levels", [])
        if start_level_name not in levels:
            return

        start_idx = levels.index(start_level_name)
        levels_to_clean = levels[start_idx:]

        for lvl in levels_to_clean:
            log_dir = os.path.join(self.log_dir_base, lvl)
            if os.path.exists(log_dir):
                logger.info(f"Cleaning up logs for level '{lvl}': {log_dir}")
                try:
                    shutil.rmtree(log_dir)
                except Exception as e:
                    logger.error(f"Failed to delete log directory {log_dir}: {e}")

    def _start_tensorboard(self):
        """
        Starts TensorBoard in the background, pointing to the logs directory.
        """
        logger.info(f"Starting TensorBoard on log directory: {self.log_dir_base}")
        try:
            # On Windows, we use 'start' or 'python -m tensorboard.main'
            # 'tensorboard' might not be in PATH but 'python -m tensorboard.main' usually works if installed
            # Using Popen to run it in background
            cmd = ["python", "-m", "tensorboard.main", "--logdir", self.log_dir_base]
            
            # Use DETACHED_PROCESS on Windows to ensure it lives after script exits if needed, 
            # but usually just Popen is enough for concurrent execution.
            # redirect stdout/stderr to devnull to avoid cluttering training logs
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
            logger.info("TensorBoard process launched in background (default port 6006).")
        except Exception as e:
            logger.error(f"Failed to start TensorBoard: {e}")

    def _train_level(self, level_name, model_path_to_load=None, reset_timesteps=True):
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
        
        # Resolve Policy Kwargs strings to classes
        if "policy_kwargs" in model_params:
            pk = model_params["policy_kwargs"]
            if "features_extractor_class" in pk and isinstance(pk["features_extractor_class"], str):
                class_path = pk["features_extractor_class"]
                if class_path == "src.agent.models.unit_control_network.UnitControlNetwork":
                    pk["features_extractor_class"] = UnitControlNetwork
                else:
                    # Generic resolver could be added here
                    pass

        # Resolve Environment Class
        env_class_name = env_params.get("env_class", "Level0Env")

        # Use MaskableSubprocVecEnv for parallel training if num_envs > 1 and headless
        if not headless:
            logger.warning("Running in non-headless mode. Forcing num_envs=1.")
            num_envs = 1
            vec_env_cls = MaskableSyncVecEnv
        elif num_envs > 1:
            vec_env_cls = MaskableSubprocVecEnv
        else:
            vec_env_cls = MaskableSyncVecEnv
        
        logger.info(f"Creating {num_envs} environment(s) for level '{level_name}' using {vec_env_cls.__name__}...")

        EnvClass = ENV_REGISTRY.get(env_class_name)
        
        if EnvClass is None:
            # Attempt to dynamically import if not in registry (Optional, but good for extensibility)
            # For now, just log and fallback or raise
            logger.error(f"Environment class '{env_class_name}' not found in registry. Available: {list(ENV_REGISTRY.keys())}")
            raise ValueError(f"Unknown environment class: {env_class_name}")

        logger.info(f"Selected Environment Class: {EnvClass.__name__}")

        # Combined initial steps for curriculum (extract from checkpoint name if possible)
        initial_steps = 0
        if model_path_to_load:
            # Check for pattern zerg_X_steps.zip
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

        # Combine training and env parameters for the environment's view
        combined_conf = training_params.copy()
        combined_conf.update(env_params)
        combined_conf["is_training"] = True # Enable curriculum progress

        # For SubprocVecEnv, we need to pass a list of callables that return the env
        env_fns = [
            EnvFactory(
                EnvClass, level_name, not headless, env_params["grid_size"], i, num_envs,
                self.sc2_process_cls, self.client_cls, combined_conf, env_params.get("max_steps", 300),
                initial_steps=initial_steps
            )
            for i in range(num_envs)
        ]
        
        if vec_env_cls == MaskableSyncVecEnv:
             env = vec_env_cls(env_fns)
        else:
             # SubprocVecEnv also expects list of functions
             # On Windows, strictly use 'spawn'
             env = vec_env_cls(env_fns, start_method='spawn')

        try:
            # 3. Setup or Load Model
            algo_name = model_params.get("algorithm", "MaskablePPO")
            AlgoClass = MaskableRecurrentPPO if algo_name == "MaskableRecurrentPPO" else MaskablePPO
            logger.info(f"Using algorithm: {algo_name}")

            if model_path_to_load:
                logger.info(f"Resuming/Loading training from: {model_path_to_load}")
                
                # Check for space mismatches
                # We use AlgoClass to load, but if the saved model was different, it might fail or warn.
                # SB3 usually handles loading into a different class if they are compatible enough, 
                # but MaskablePPO and MaskableRecurrentPPO have different policy structures.
                
                # For safety, we try to load with the requested AlgoClass.
                try:
                    saved_model = AlgoClass.load(model_path_to_load, device="cpu", custom_objects={"verbose": 1})
                except Exception as e:
                    from src.agent.maskable_ppo import MaskablePPO as CustomMaskablePPO
                    logger.info(f"Using CustomMaskablePPO: {CustomMaskablePPO}")
                    logger.warning(f"Failed to load with {algo_name}, attempting fallback load with MaskablePPO: {e}")
                    saved_model = CustomMaskablePPO.load(model_path_to_load, device="cpu", custom_objects={"verbose": 1})

                saved_policy = saved_model.policy
                
                current_obs_shape = env.observation_space.shape
                saved_obs_shape = saved_policy.observation_space.shape
                
                obs_mismatch = current_obs_shape != saved_obs_shape
                act_mismatch = env.action_space.n != saved_policy.action_space.n
                
                # Arch mismatch: if we want recurrent but saved is not, or vice versa
                arch_mismatch = (algo_name == "MaskableRecurrentPPO" and not isinstance(saved_model, MaskableRecurrentPPO)) or \
                                (algo_name == "MaskablePPO" and isinstance(saved_model, MaskableRecurrentPPO))

                if obs_mismatch or act_mismatch or arch_mismatch:
                    logger.warning(f"Mismatch detected. Obs: {saved_obs_shape} -> {current_obs_shape}, Act: {saved_policy.action_space.n} -> {env.action_space.n}, Arch Change: {arch_mismatch}")
                    logger.info("Transferring weights to a new model structure.")
                    
                    state_dict = saved_policy.state_dict()
                    
                    # 1. Create fresh model first to get target state dict for smart grafting
                    model_kwargs = {
                        "policy": model_params["policy"],
                        "env": env,
                        "verbose": int(training_params.get("verbose", 1)),
                        "tensorboard_log": log_dir,
                        "learning_rate": training_params["learning_rate"],
                        "batch_size": training_params["batch_size"],
                        "n_steps": training_params["n_steps"],
                        "gamma": model_params["gamma"],
                        "ent_coef": model_params["ent_coef"],
                        "n_epochs": model_params["n_epochs"],
                        "policy_kwargs": {**model_params.get("policy_kwargs", {}), "normalize_images": False},
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        # Imitation Learning
                        "imitation_data_percentage": training_params.get("imitation_data_percentage", 0.0),
                        "imitation_data_max_steps": training_params.get("imitation_data_max_steps", 0),
                        "imitation_coef": training_params.get("imitation_coef", 0.5),
                        "level_name": level_name,
                    }
                    if AlgoClass == MaskableRecurrentPPO:
                        model_kwargs["rnn_type"] = model_params.get("rnn_type", "lstm")

                    model = AlgoClass(**model_kwargs)
                    target_state_dict = model.policy.state_dict()

                    # 2. Handle Mismatches via Smart Grafting
                    logger.info(f"Performing smart weight transfer (Inputs: {current_obs_shape[0]} ch, Actions: {env.action_space.n}, Grid: {env_params['grid_size']})...")
                    try:
                        state_dict = graft_model_weights(
                            state_dict, 
                            new_in_channels=current_obs_shape[0],
                            new_action_dim=env.action_space.n,
                            new_grid_size=env_params["grid_size"],
                            target_state_dict=target_state_dict
                        )
                    except Exception as e:
                        logger.error(f"Weight grafting failed: {e}")
                        raise e
                    
                    # 3. Load compatible weights
                    missing, unexpected = model.policy.load_state_dict(state_dict, strict=False)
                    logger.info(f"Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    
                else:
                    custom_objects = {
                        "learning_rate": training_params["learning_rate"],
                        "n_steps": training_params["n_steps"],
                        "batch_size": training_params["batch_size"],
                        "imitation_data_percentage": training_params.get("imitation_data_percentage", 0.0),
                        "imitation_data_max_steps": training_params.get("imitation_data_max_steps", 0),
                        "imitation_coef": training_params.get("imitation_coef", 0.5),
                        "level_name": level_name,
                    }
                    model = AlgoClass.load(
                        model_path_to_load,
                        env=env,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        tensorboard_log=log_dir,
                        custom_objects=custom_objects
                    )
                    # Re-initialize imitation buffer as .load() bypasses constructor
                    if hasattr(model, "_setup_imitation"):
                        model._setup_imitation(
                            level_name=level_name,
                            imitation_data_percentage=training_params.get("imitation_data_percentage", 0.0),
                            imitation_data_max_steps=training_params.get("imitation_data_max_steps", 0),
                            imitation_coef=training_params.get("imitation_coef", 0.5)
                        )
            else:
                logger.info(f"Starting new training session with {algo_name}.")
                model_kwargs = {
                    "policy": model_params["policy"],
                    "env": env,
                    "verbose": 1,
                    "tensorboard_log": log_dir,
                    "learning_rate": training_params["learning_rate"],
                    "batch_size": training_params["batch_size"],
                    "n_steps": training_params["n_steps"],
                    "gamma": model_params["gamma"],
                    "ent_coef": model_params["ent_coef"],
                    "n_epochs": model_params["n_epochs"],
                    "policy_kwargs": {**model_params.get("policy_kwargs", {}), "normalize_images": False},
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    # Imitation Learning
                    "imitation_data_percentage": training_params.get("imitation_data_percentage", 0.0),
                    "imitation_data_max_steps": training_params.get("imitation_data_max_steps", 0),
                    "imitation_coef": training_params.get("imitation_coef", 0.5),
                    "level_name": level_name,
                }
                if AlgoClass == MaskableRecurrentPPO:
                    model_kwargs["rnn_type"] = model_params.get("rnn_type", "lstm")

                model = AlgoClass(**model_kwargs)

            # 4. Train Model
            # Calculate steps to train
            total_steps_target = training_params["training_steps"]
            steps_to_train = total_steps_target
            
            if model_path_to_load and not reset_timesteps:
                # If resuming, we need to account for already trained steps
                # Note: model.num_timesteps is reliable here because we haven't trained yet in this session
                steps_already_done = model.num_timesteps
                steps_to_train = total_steps_target - steps_already_done
                logger.info(f"Resuming training. Target: {total_steps_target}, Already done: {steps_already_done}, Remaining: {steps_to_train}")
                
                if steps_to_train <= 0:
                    logger.info("Training already reached target steps. Saving and finishing.")
                    final_model_path = os.path.join(model_dir, "final_model.zip")
                    model.save(final_model_path)
                    return final_model_path
            
            logger.info(f"Starting training loop for {steps_to_train} timesteps (Reset timesteps: {reset_timesteps})...")
            
            # CRITICAL: Clear the episode info buffer when starting a new level
            # This prevents metrics from the previous level (e.g., short episode lengths)
            # from leaking into the new level and triggering early stopping callbacks.
            if hasattr(model, "ep_info_buffer") and model.ep_info_buffer is not None:
                model.ep_info_buffer.clear()
            
            # Watchdog to detect hangs in the training loop
            # Default to 1 hour (3600s) to accommodate slow CPU training optimization phases.
            w_timeout = training_params.get("watchdog_timeout", 3600.0)
            watchdog = Watchdog(timeout=w_timeout, recurring=True)
            watchdog.start()

            # Calculate save frequency to be exactly 50,000 global steps
            # save_freq in SB3 is per vectorized step (n_calls)
            save_freq_steps = 50000
            save_freq_calls = max(1, save_freq_steps // num_envs)
            
            checkpoint_callback = CheckpointCallback(save_freq=save_freq_calls, save_path=model_dir, name_prefix=level_name)

            diagnostics_callback = DiagnosticsCallback(check_freq=1024, watchdog=watchdog)
            
            callback_list = [checkpoint_callback, diagnostics_callback]

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

            try:
                model.learn(
                    total_timesteps=steps_to_train,
                    callback=CallbackList(callback_list),
                    reset_num_timesteps=reset_timesteps
                )
            except (EOFError, BrokenPipeError) as e:
                logger.error(f"FATAL: Worker process crashed during training of level '{level_name}'.")
                logger.error(f"Error detail: {e}")
                logger.error("Please check the worker logs in 'logs/worker_*.log' for the root cause (e.g. SC2 crash, timeout, or Python error).")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error during training: {e}")
                raise e

            # 5. Save Final Model
            logger.info("Training for this level complete. Saving final model.")
            final_model_path = os.path.join(model_dir, "final_model.zip")
            model.save(final_model_path)
            return final_model_path

        finally:
            # 6. Clean Up
            if 'watchdog' in locals():
                watchdog.stop()
            logger.info("Closing environments for this level.")
            env.close()
        logger.info(f"--- Finished Level: {level_name} ---")


    def run(self):
        """Executes the entire training pipeline."""
        logger.info(f"Starting training pipeline for levels: {self.pipeline_config['levels']}")
        
        # Ensure clean state before starting (opt-out via global_sc2_kill_on_boundaries)
        if self.global_sc2_kill_on_boundaries:
            kill_switch_sc2()
        cleanup_worker_logs()
        
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
            reset_timesteps = True

            if status == "CONTINUE_TRAINING":
                model_to_load = model_path # Continue from the checkpoint
                reset_timesteps = False # Keep the step count
            elif status == "NEEDS_TRAINING":
                # If it's a new training, try to load the model from the previous level
                model_to_load = last_final_model_path
                reset_timesteps = True # Start fresh for this level
                
                # Cleanup logs for this level and all subsequent ones
                self._cleanup_logs(level_name)
            
            # Train the level
            final_model_path = self._train_level(level_name, model_path_to_load=model_to_load, reset_timesteps=reset_timesteps)
            last_final_model_path = final_model_path

        logger.info("Training pipeline finished.")
        if self.global_sc2_kill_on_boundaries:
            kill_switch_sc2()


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
import errno
import os
import sys
import time
from collections import deque
import torch
import numpy as np
import yaml
import json
import gc
from loguru import logger
import multiprocessing as mp
from gymnasium import spaces

# Add project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_PROJECT_ROOT)

from train.train_universal import UniversalPipelineManager, BOT_REGISTRY
from src.sc2_bot_universal import SC2BotUniversal
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.maskable_ppo import MaskablePPO
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.universal_policy import UniversalActorCriticPolicy
from src.agent.models.unit_control_network import UnitControlNetwork
from src.utils.training_device import resolve_training_device, warmup_cuda_if_needed
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer

import shutil
import psutil

# Async buffer: memmaps + payload.npz + hot-swap model zip every rollout — heavy disk I/O.
# Default to C:/temp (typically the OS SSD) for lowest latency.
# Override with env var SC2_ASYNC_BUFFER_DIR if you want a different location.
# Example: $env:SC2_ASYNC_BUFFER_DIR = "D:/sc2_buffer"
_DEFAULT_ASYNC_DIR = "C:/temp/sc2_async_buffer"
TEMP_DIR = os.path.normpath(os.environ.get("SC2_ASYNC_BUFFER_DIR", _DEFAULT_ASYNC_DIR))
# Collector loads from this path. Never read `latest_learner_model.zip` directly — the learner
# replaces that file; simultaneous open on Windows causes Errno 13 / sharing violations.
COLLECTOR_WEIGHTS_STAGING_ZIP = os.path.join(TEMP_DIR, "latest_learner_model_collector_staging.zip")


def force_kill_sc2():
    """Aggressively kills any SC2 and related processes at startup."""
    logger.info("Performing hard cleanup of StarCraft II processes...")
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name'].lower()
            if "starcraft" in name or "sc2" in name or "battle.net" in name:
                logger.debug(f"Killing process: {name} ({proc.pid})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def clear_temp_dir():
    """Wipes the async buffer directory entirely."""
    if os.path.exists(TEMP_DIR):
        logger.info(f"Cleaning up async buffer directory: {TEMP_DIR}")
        try:
            # We use a retry loop because Windows sometimes locks files
            for _ in range(3):
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
                if not os.path.exists(TEMP_DIR): break
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Startup cleanup failed: {e}")
    os.makedirs(TEMP_DIR, exist_ok=True)


def _is_transient_file_lock(err: OSError) -> bool:
    """True when another process (e.g. collector loading the zip) still holds the file."""
    if isinstance(err, PermissionError):
        return True
    win = getattr(err, "winerror", None)
    if win == 32:  # ERROR_SHARING_VIOLATION
        return True
    return err.errno in (errno.EACCES, errno.EPERM, errno.EBUSY)


def replace_file_retry(src: str, dst: str, *, retries: int = 50, delay_sec: float = 0.1) -> None:
    """
    Atomic replace (os.replace). Retries on Windows sharing violations (e.g. antivirus
    or a reader still closing a handle on dst).
    """
    last: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                if attempt == 0 or attempt % 10 == 9:
                    logger.debug(
                        f"replace_file_retry: dst busy ({dst}), attempt {attempt + 1}/{retries}"
                    )
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


def copy_file_retry(src: str, dst: str, *, retries: int = 50, delay_sec: float = 0.1) -> None:
    """
    Copy a file with retries on Windows sharing / antivirus locks (Errno 13, WinError 32).
    Overwrites dst when the copy succeeds.
    """
    last: OSError | None = None
    for attempt in range(retries):
        try:
            shutil.copyfile(src, dst)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                if attempt == 0 or attempt % 10 == 9:
                    logger.debug(
                        f"copy_file_retry: busy ({src!r} -> {dst!r}), attempt {attempt + 1}/{retries}"
                    )
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


def stage_learner_zip_for_collector(src_zip: str, staging_zip: str) -> None:
    """
    Atomically publish a private copy for the collector: copy to *.part then replace staging.
    Avoids reading a zip the learner is still writing or replacing.
    """
    part = staging_zip + ".part"
    copy_file_retry(src_zip, part)
    replace_file_retry(part, staging_zip)


def set_parameters_with_retry(model, load_path: str, device: torch.device, *, retries: int = 12, delay_sec: float = 0.12) -> None:
    """SB3 set_parameters can still hit brief locks (AV); retry transient OS errors."""
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            model.set_parameters(load_path, device=device)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                if attempt == 0 or attempt % 4 == 3:
                    logger.debug(
                        f"set_parameters_with_retry: {load_path!r} busy, attempt {attempt + 1}/{retries}"
                    )
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


def remove_file_retry(path: str, *, retries: int = 50, delay_sec: float = 0.1) -> None:
    """Best-effort unlink with retries (Windows antivirus / other readers)."""
    if not os.path.exists(path):
        return
    last: OSError | None = None
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except OSError as e:
            last = e
            if _is_transient_file_lock(e):
                time.sleep(delay_sec)
                continue
            raise
    assert last is not None
    raise last


class AsyncCollector(mp.Process):
    """
    PRODUCER: Plays StarCraft II and streams data to the disk buffer.
    It never stops unless the Learner is too slow.
    """
    def __init__(
        self,
        level_name,
        config,
        stop_event,
        model_path=None,
        resume_timesteps_from_save=False,
    ):
        super().__init__()
        self.level_name = level_name
        self.config = config
        self.stop_event = stop_event
        self.model_path = model_path
        # True only for same-level checkpoint resume; False for fresh start or curriculum (prior level weights).
        self.resume_timesteps_from_save = resume_timesteps_from_save

    def run(self):
        logger.info(f"Collector started for level: {self.level_name}")
        # Initialize env and model using UniversalPipelineManager logic
        manager = UniversalPipelineManager(config_dir=self.config['config_dir'])
        
        # 1. Setup Env (Simplified from _train_level)
        training_params = self.config["training"]
        train_device = resolve_training_device(training_params)
        warmup_cuda_if_needed(train_device)
        logger.info(f"Collector policy device: {train_device}")
        env_params = self.config["env"]
        num_envs = training_params["num_envs"]
        
        # Use the bot specified in the environment or registry
        # DO NOT force SC2BotUniversal here, as it breaks multi-unit levels
        
        # Use existing factory to build env
        from train.train_pipeline import MaskableSubprocVecEnv, MaskableSyncVecEnv
        from train.env_utils import EnvFactory
        from train.train_pipeline import ENV_REGISTRY
        
        EnvClass = ENV_REGISTRY.get(env_params.get("env_class", "Level0Env"))
        # Default 32 matches universal MultiDiscrete spatial resolution; key optional in YAML.
        _grid = env_params.get("grid_size", 32)
        env_fns = [
            EnvFactory(
                EnvClass, self.level_name, False, _grid, i, num_envs,
                None, None, self.config, env_params.get("max_steps", 300)
            )
            for i in range(num_envs)
        ]
        vec_env = (
            MaskableSubprocVecEnv(env_fns, start_method="spawn")
            if num_envs > 1
            else MaskableSyncVecEnv(env_fns)
        )

        # 2. Setup Policy
        model_params = self.config.get("model", {})
        algo_name = model_params.get("algorithm", "MaskablePPO")
        rnn_type = model_params.get("rnn_type", "gru")
        hidden_size = model_params.get("policy_kwargs", {}).get("lstm_hidden_size", 512)
        
        AlgoClass = MaskableRecurrentPPO if algo_name == "MaskableRecurrentPPO" else MaskablePPO
        PolicyClass = UniversalRecurrentPolicy if AlgoClass == MaskableRecurrentPPO else UniversalActorCriticPolicy
        
        policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
        policy_kwargs["spatial_resolution"] = 32
        # Only recurrent policies accept/need `rnn_type`. For plain PPO we must not
        # inject it into sb3_contrib's MaskableActorCriticPolicy (non-recurrent).
        if AlgoClass == MaskableRecurrentPPO:
            policy_kwargs["rnn_type"] = rnn_type # CRITICAL: SB3 compares this with stored kwargs
        if "features_extractor_class" in policy_kwargs and policy_kwargs["features_extractor_class"] == "src.agent.models.unit_control_network.UnitControlNetwork":
            policy_kwargs["features_extractor_class"] = UnitControlNetwork

        algo_kwargs = {
            "policy": PolicyClass,
            "env": vec_env,
            "policy_kwargs": policy_kwargs,
            "n_steps": training_params["n_steps"],
            "batch_size": training_params["batch_size"],
            "gamma": model_params["gamma"],
            "device": train_device,
        }
        if AlgoClass == MaskableRecurrentPPO:
            algo_kwargs["rnn_type"] = rnn_type

        if self.model_path:
            logger.info(f"Loading weights from: {self.model_path}")
            load_kwargs = {
                "env": vec_env,
                "policy_kwargs": policy_kwargs,
                "device": train_device,
            }
            if AlgoClass == MaskableRecurrentPPO:
                load_kwargs["rnn_type"] = rnn_type
            try:
                model = AlgoClass.load(self.model_path, **load_kwargs)
            except Exception as e:
                logger.warning(
                    f"Could not load weights from '{self.model_path}' "
                    f"(likely architecture change between levels): {e}. Starting fresh."
                )
                model = AlgoClass(**algo_kwargs)
        else:
            model = AlgoClass(**algo_kwargs)

        if self.model_path and not self.resume_timesteps_from_save:
            model.num_timesteps = 0

        fe = model.policy.features_extractor
        if hasattr(fe, "cnn_trunk"):
            c0 = fe.cnn_trunk[0]
            logger.info(f"Collector Model initialized (global CNN in_channels={c0.weight.shape[1]})")
        elif hasattr(fe, "cnn") and len(fe.cnn) > 0:
            logger.info(f"Collector Model initialized with CNN input shape: {fe.cnn[0].weight.shape}")
        else:
            logger.info("Collector Model initialized (features extractor ready)")
            
        # Save initial weights for Learner to ensure perfect synchronization at start
        initial_path = os.path.join(TEMP_DIR, "initial_model.zip")
        model.save(initial_path)
        logger.info("Collector: Saved initial weights for Learner synchronization.")
        
        # 3. Static Double-Buffer Initialization
        # We manually create fixed directories buffer_0 and buffer_1
        buffer_idx = 0
        buffers = []
        for i in range(2):
            static_path = os.path.join(TEMP_DIR, f"buffer_{i}")
            # We must instantiate the buffer to create the structure
            buf = AsyncDiskBuffer(
                training_params["n_steps"], 
                vec_env.observation_space, 
                vec_env.action_space, 
                n_envs=num_envs, 
                temp_dir=TEMP_DIR, 
                rnn_type=rnn_type, 
                hidden_size=hidden_size
            )
            # Force the instance dir to be static
            # Cleanup old one if needed
            if os.path.exists(buf.instance_dir):
                try: os.rmdir(buf.instance_dir) 
                except: pass
            
            # Re-point to static dir
            buf.instance_id = f"buffer_{i}"
            buf.instance_dir = static_path
            os.makedirs(buf.instance_dir, exist_ok=True)
            
            # Re-initialize memmaps in the new static location
            buf.disk_files = {}
            for key, subspace in vec_env.observation_space.spaces.items():
                if "map" in key or "tactical" in key:
                    filename = os.path.join(buf.instance_dir, f"obs_{key}.bin")
                    shape = (buf.buffer_size, buf.n_envs, *subspace.shape)
                    # Close old map if exists from init
                    if key in buf.observations and isinstance(buf.observations[key], np.memmap):
                        buf.observations[key]._mmap.close()
                    
                    # Create new static map
                    buf.observations[key] = np.memmap(filename, dtype=np.float32, mode='w+', shape=shape)
                    buf.disk_files[key] = filename
            
            buffers.append(buf)

        logger.info("Collector entering main loop...")
        obs = vec_env.reset()
        
        # Track metrics manually to avoid wrapper attribute errors
        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs)
        recent_ep_rewards = []
        recent_ep_lengths = []
        
        while not self.stop_event.is_set():
            current_buffer = buffers[buffer_idx % 2]
            
            # Ensure learner is done with this buffer before overwriting
            consume_timeout = training_params.get("buffer_consumer_timeout", 300)
            if not current_buffer.wait_for_consumer(timeout=consume_timeout):
                logger.error("Learner hung? Collector pausing.")
                time.sleep(5)
                continue

            # Check for updated model weights from Learner
            updated_model_path = os.path.join(TEMP_DIR, "latest_learner_model.zip")
            if os.path.exists(updated_model_path):
                try:
                    # Atomic check: only load if it's new
                    mtime = os.path.getmtime(updated_model_path)
                    if not hasattr(self, 'last_model_load_time') or mtime > self.last_model_load_time:
                        # Stage to a private zip: opening latest_learner_model.zip while the learner
                        # replaces it (or AV scans it) causes Permission denied on Windows.
                        logger.info(f"Collector found new weights (mtime={mtime}). Staging copy...")
                        stage_learner_zip_for_collector(updated_model_path, COLLECTOR_WEIGHTS_STAGING_ZIP)
                        set_parameters_with_retry(model, COLLECTOR_WEIGHTS_STAGING_ZIP, model.device)
                        self.last_model_load_time = mtime
                        logger.info("Collector: Successfully loaded fresh weights from Learner.")
                except Exception as e:
                    logger.error(f"Collector failed to load weights: {e}")

            # FILL BUFFER
            current_buffer.reset()
            # Use class-based detection instead of attribute presence:
            # some model objects may carry compatibility attributes even when
            # running non-recurrent PPO.
            is_recurrent = isinstance(model, MaskableRecurrentPPO)
            for step in range(training_params["n_steps"]):
                # Get actions, values, and log_probs from model
                # We need torch tensors for the policy.forward call
                with torch.no_grad():
                    obs_tensor = model.policy.obs_to_tensor(obs)[0]
                    # We assume episode_starts are False initially or tracked
                    # Match SB3 OnPolicyAlgorithm: episode_starts for an observation is True on the
                    # first step after reset; initial value must be ones (not zeros).
                    if not hasattr(self, "last_dones"):
                        self.last_dones = np.ones(num_envs, dtype=bool)
                    masks = np.array(vec_env.env_method("action_masks"))
                    masks_tensor = torch.as_tensor(masks).to(model.device)

                    # Call forward to get EVERYTHING we need for the buffer
                    if is_recurrent:
                        starts_tensor = torch.as_tensor(self.last_dones).to(model.device).float()
                        actions_tensor, values_tensor, log_probs_tensor, new_lstm_states = model.policy.forward(
                            obs_tensor, model._last_lstm_states, starts_tensor, action_masks=masks_tensor
                        )
                    else:
                        actions_tensor, values_tensor, log_probs_tensor = model.policy.forward(
                            obs_tensor, deterministic=False, action_masks=masks_tensor
                        )
                        new_lstm_states = None
                    
                    actions = actions_tensor.cpu().numpy()
                    values = values_tensor.cpu().numpy()
                    log_probs = log_probs_tensor.cpu().numpy()

                # Step env
                new_obs, rewards, dones, infos = vec_env.step(actions)
                
                # Update manual metrics
                episode_rewards += rewards
                episode_lengths += 1
                
                # Buffer for custom info metrics (success rates, distance, etc.)
                # deque(maxlen=) avoids O(n) list.pop(0) churn as training runs for hours.
                if not hasattr(self, 'custom_metrics'):
                    self.custom_metrics = {}

                for i, done in enumerate(dones):
                    # Capture metrics from the info dict (e.g., success, etc.)
                    info = infos[i]
                    for key, val in info.items():
                        if isinstance(val, (int, float, np.number)):
                            if key not in self.custom_metrics:
                                self.custom_metrics[key] = deque(maxlen=100)
                            self.custom_metrics[key].append(val)

                    if done:
                        recent_ep_rewards.append(episode_rewards[i])
                        recent_ep_lengths.append(episode_lengths[i])
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0
                        # Keep only last 100 episodes
                        if len(recent_ep_rewards) > 100:
                            recent_ep_rewards.pop(0)
                            recent_ep_lengths.pop(0)

                # Add to buffer with ALL required fields
                current_buffer.add(
                    obs, 
                    actions, 
                    rewards, 
                    self.last_dones, 
                    values_tensor, 
                    log_probs_tensor, 
                    model._last_lstm_states if is_recurrent else None,
                    masks
                )
                
                obs = new_obs
                self.last_dones = dones
                if is_recurrent:
                    model._last_lstm_states = new_lstm_states

            # CRITICAL: Compute GAE (Advantages & Returns) before sending to Learner
            # We need the value of the LAST observation to bootstrap
            with torch.no_grad():
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                masks = np.array(vec_env.env_method("action_masks"))
                masks_tensor = torch.as_tensor(masks).to(model.device)
                
                # We only need the value
                if is_recurrent:
                    starts_tensor = torch.as_tensor(dones).to(model.device).float()
                    last_values = model.policy.predict_values(obs_tensor, model._last_lstm_states.vf, starts_tensor)
                else:
                    last_values = model.policy.predict_values(obs_tensor)
            
            current_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

            # FLUSH MEMMAPS TO DISK
            # Ensure the Learner sees the data we just wrote
            for key, mm in current_buffer.observations.items():
                if isinstance(mm, np.memmap): mm.flush()
            
            # SIGNAL LEARNER
            signal_meta = {
                "level": self.level_name, 
                "steps": training_params["n_steps"]
            }
            
            if recent_ep_rewards:
                signal_meta["rollout/ep_rew_mean"] = np.mean(recent_ep_rewards)
                signal_meta["rollout/ep_len_mean"] = np.mean(recent_ep_lengths)
            
            # Add all other captured metrics
            if hasattr(self, 'custom_metrics'):
                for key, vals in self.custom_metrics.items():
                    if vals:
                        # SB3 standard is to put environment metrics in 'rollout/'
                        # but we'll preserve the original key names or prefix them
                        prefix = "" if "/" in key else "env/"
                        signal_meta[f"{prefix}{key}"] = np.mean(vals)

            current_buffer.signal_ready(signal_meta)
            buffer_idx += 1

        vec_env.close()

class AsyncLearner:
    """
    CONSUMER: Watches for ready buffers and performs GPU updates.
    """
    def __init__(self, level_name, config, resume_timesteps_from_save=False):
        self.level_name = level_name
        self.config = config
        self.resume_timesteps_from_save = resume_timesteps_from_save
        self.latest_model_path = os.path.join(TEMP_DIR, "latest_learner_model.zip")
        training_params = config.get("training", {})
        tb_subdir = training_params.get("tensorboard_subdir", "async")
        self.log_dir = os.path.join("logs_universal", level_name, tb_subdir)
        os.makedirs(self.log_dir, exist_ok=True)
        # Ensure model directory exists
        self.model_dir = os.path.join("models_universal", level_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.save_freq = config["training"].get("save_freq", 50000)

    def close(self):
        """Clean up handles to prevent WinError 32 leaks across curriculum levels"""
        if hasattr(self, 'logger'):
            try:
                self.logger.close()
            except Exception as e:
                logger.warning(f"Could not close logger: {e}")
        if hasattr(self, 'learner_buffers'):
            for buf in self.learner_buffers.values():
                if hasattr(buf, 'close_memmaps'):
                    buf.close_memmaps()

    def run(self):
        logger.info(f"Learner started for level: {self.level_name}")
        training_params = self.config["training"]
        train_device = resolve_training_device(training_params)
        warmup_cuda_if_needed(train_device)
        logger.info(
            f"Learner training device: {train_device} "
            f"(torch.cuda.is_available()={torch.cuda.is_available()})"
        )
        model_params = self.config.get("model", {})
        
        # 1. Setup Dummy Env for Model
        from train.env_utils import create_dummy_env_from_config
        dummy_env = create_dummy_env_from_config(self.config)
        
        algo_name = model_params.get("algorithm", "MaskablePPO")
        rnn_type = model_params.get("rnn_type", "gru")
        hidden_size = model_params.get("policy_kwargs", {}).get("lstm_hidden_size", 512)
        
        AlgoClass = MaskableRecurrentPPO if algo_name == "MaskableRecurrentPPO" else MaskablePPO
        PolicyClass = UniversalRecurrentPolicy if AlgoClass == MaskableRecurrentPPO else UniversalActorCriticPolicy
        
        policy_kwargs = {**model_params.get("policy_kwargs", {}), "normalize_images": False}
        policy_kwargs["spatial_resolution"] = 32
        if "features_extractor_class" in policy_kwargs and policy_kwargs["features_extractor_class"] == "src.agent.models.unit_control_network.UnitControlNetwork":
            policy_kwargs["features_extractor_class"] = UnitControlNetwork

        algo_kwargs = {
            "policy": PolicyClass,
            "env": dummy_env,
            "n_steps": training_params["n_steps"],
            "batch_size": training_params["batch_size"],
            "n_epochs": model_params.get("n_epochs", 4),
            "gamma": model_params.get("gamma", 0.99),
            "learning_rate": training_params["learning_rate"],
            "ent_coef": model_params.get("ent_coef", 0.0),
            "policy_kwargs": policy_kwargs,
            "device": train_device,
            "tensorboard_log": self.log_dir,
            # Default None: run full n_epochs (sync universal also omits this). Hardcoded 0.015 caused
            # early exit on "easy" rollouts → far fewer minibatches vs hard settings. Override in YAML:
            # model: { target_kl: 0.015 }
            "target_kl": model_params.get("target_kl"),
        }
        if AlgoClass == MaskableRecurrentPPO:
            algo_kwargs["rnn_type"] = rnn_type
            
        tk = algo_kwargs.get("target_kl")
        logger.info(
            f"Learner PPO target_kl={tk!r} (approx_kl early-stop at {1.5 * tk:.4f} per batch when set)"
            if tk is not None
            else "Learner PPO target_kl=None (no KL early-stop; full n_epochs each update)"
        )
        model = AlgoClass(**algo_kwargs)
        model.learner_profile = training_params.get("learner_profile", True)
        p0 = next(model.policy.parameters())
        logger.info(f"Learner model.device={model.device} policy params on {p0.device}")

        fe = model.policy.features_extractor
        if hasattr(fe, "cnn_trunk"):
            c0 = fe.cnn_trunk[0]
            logger.info(f"Learner Model initialized (global CNN in_channels={c0.weight.shape[1]})")
        elif hasattr(fe, "cnn") and len(fe.cnn) > 0:
            logger.info(f"Learner Model initialized with CNN input shape: {fe.cnn[0].weight.shape}")
        else:
            logger.info("Learner Model initialized (features extractor ready)")
        
        # WAIT FOR COLLECTOR'S INITIAL WEIGHTS
        initial_path = os.path.join(TEMP_DIR, "initial_model.zip")
        logger.info("Learner: Waiting for initial weights from Collector...")
        total_steps_trained = 0
        for _ in range(60): # Wait up to 60 seconds
            if os.path.exists(initial_path):
                try:
                    model.set_parameters(initial_path, device=train_device)

                    import zipfile
                    with zipfile.ZipFile(initial_path, 'r') as archive:
                        if self.resume_timesteps_from_save and 'data' in archive.namelist():
                            data_content = archive.read('data').decode('utf-8')
                            model_data = json.loads(data_content)
                            total_steps_trained = model_data.get("num_timesteps", 0)
                            logger.info(f"Learner: Resuming from {total_steps_trained} timesteps.")
                        else:
                            total_steps_trained = 0
                            model.num_timesteps = 0
                            logger.info(
                                "Learner: Timestep counter at 0 for this level "
                                "(matches sync train_universal reset_num_timesteps on curriculum / fresh start)."
                            )

                    logger.info("Learner: Synchronized initial weights with Collector.")
                    break
                except Exception as e:
                    logger.warning(f"Learner failed to load initial weights: {e}")
            time.sleep(1)
        
        # Configure model logger
        from stable_baselines3.common.logger import configure
        new_logger = configure(self.log_dir, ["stdout", "tensorboard", "csv"])
        model.set_logger(new_logger)
        self.logger = new_logger
        logger.info(f"Learner logging configured to: {self.log_dir}")

        # PRE-INITIALIZE LEARNER BUFFERS
        # We create persistent buffer objects that point to the static directories.
        # This prevents opening/closing file handles during the loop.
        self.learner_buffers = {}
        for i in range(2):
            instance_id = f"buffer_{i}"
            instance_dir = os.path.join(TEMP_DIR, instance_id)
            
            # Wait for collector to create the structure if it doesn't exist yet
            # (Collector creates them at startup)
            logger.info(f"Learner initializing static buffer: {instance_id}")
            
            # We construct the buffer manually to ensure it points to the right place
            # without creating NEW subdirectories
            buf = AsyncDiskBuffer(
                buffer_size=training_params["n_steps"],
                observation_space=dummy_env.observation_space,
                action_space=dummy_env.action_space,
                n_envs=training_params["num_envs"],
                temp_dir=TEMP_DIR,
                rnn_type=rnn_type,
                hidden_size=hidden_size
            )
            
            # Force re-point to the static directory 
            # (AsyncDiskBuffer init creates a random dir, we delete it and swap)
            try: os.rmdir(buf.instance_dir)
            except: pass
            
            buf.instance_id = instance_id
            buf.instance_dir = instance_dir
            
            # Close and remove dummy memmaps created by AsyncDiskBuffer.__init__
            buf.disk_files = {}
            for key in list(buf.observations.keys()):
                if isinstance(buf.observations[key], np.memmap):
                    try:
                        buf.observations[key]._mmap.close()
                    except:
                        pass
                    del buf.observations[key]
            
            # Placeholder to be filled when files actually exist
            self.learner_buffers[instance_id] = buf

        logger.info("Learner waiting for first buffer...")
        start_time = time.time()
        
        while True:
            # Poll all possible buffer directories for a 'ready' signal
            found_buffer = False
            for d in os.listdir(TEMP_DIR):
                instance_dir = os.path.join(TEMP_DIR, d)
                if not os.path.isdir(instance_dir): continue
                
                signal_path = os.path.join(instance_dir, "buffer_ready.json")
                if os.path.exists(signal_path):
                    logger.info(f"Learner: Consuming buffer from {d}")
                    
                    try:
                        with open(signal_path, "r") as f:
                            meta = json.load(f)
                    except:
                        continue

                    # Update internal timestep for logging
                    num_steps = meta["buffer_size"] * meta["n_envs"]
                    total_steps_trained += num_steps
                    model.num_timesteps = total_steps_trained

                    # Calculate and log FPS
                    elapsed_time = time.time() - start_time
                    fps = int(total_steps_trained / elapsed_time) if elapsed_time > 0 else 0
                    model.logger.record("time/fps", fps)
                    model.logger.record("time/total_timesteps", total_steps_trained, exclude="tensorboard")
                    
                    # 1. SWITCH TO STATIC BUFFER
                    if d not in self.learner_buffers:
                        # Should not happen with static buffering 0/1
                        logger.warning(f"Unknown buffer ID {d}, skipping")
                        continue
                        
                    target_buffer = self.learner_buffers[d]
                    
                    # LOAD RAM PAYLOAD
                    payload_path = os.path.join(instance_dir, "payload.npz")
                    if os.path.exists(payload_path):
                        try:
                            # Always close NpzFile: leaving it open leaks Windows handles and keeps
                            # the archive locked, which worsens over long runs (slowdown + WinError 32).
                            with np.load(payload_path) as payload:
                                for key in ['actions', 'rewards', 'episode_starts', 'values', 'log_probs', 'advantages', 'returns', 'hidden_states_pi', 'hidden_states_vf']:
                                    try:
                                        setattr(target_buffer, key, payload[key].copy())
                                    except Exception as e:
                                        logger.error(f"Failed on key {key}: {getattr(target_buffer, key).flags}")
                                        raise e
                                        
                                if 'cell_states_pi' in payload and payload['cell_states_pi'].size > 0:
                                    target_buffer.cell_states_pi = payload['cell_states_pi'].copy()
                                    target_buffer.cell_states_vf = payload['cell_states_vf'].copy()
                                if 'action_masks' in payload and payload['action_masks'].size > 0:
                                    try:
                                        target_buffer.action_masks = payload['action_masks'].copy()
                                    except Exception as e:
                                        logger.error(f"Failed on action_masks: {target_buffer.action_masks.flags}")
                                        raise e

                                # Load RAM-based observations (e.g., 'command')
                                for key in payload.files:
                                    if key.startswith("obs_"):
                                        obs_key = key[4:]
                                        if obs_key in target_buffer.observations:
                                            try:
                                                if isinstance(target_buffer.observations[obs_key], np.memmap):
                                                    if not target_buffer.observations[obs_key].flags.writeable:
                                                        target_buffer.observations[obs_key].flags.writeable = True
                                                    target_buffer.observations[obs_key][:] = payload[key]
                                                else:
                                                    target_buffer.observations[obs_key] = payload[key].copy()
                                            except Exception as e:
                                                logger.error(f"Failed on obs {obs_key}: {target_buffer.observations[obs_key].flags}")
                                                raise e
                        except Exception as e:
                            logger.error(f"Failed to load buffer payload: {e}")
                            continue
                    else:
                        logger.warning(f"Payload not found for {d}, skipping")
                        continue

                    # Ensure memmaps are bound (Lazy binding on first use)
                    for key in dummy_env.observation_space.keys():
                        if key in ["map", "tactical"]:
                            if key not in target_buffer.observations or not isinstance(target_buffer.observations[key], np.memmap):
                                filename = os.path.join(instance_dir, f"obs_{key}.bin")
                                if os.path.exists(filename):
                                    subspace = dummy_env.observation_space.spaces[key]
                                    shape = (meta["buffer_size"], meta["n_envs"], *subspace.shape)
                                    target_buffer.observations[key] = np.memmap(filename, dtype=np.float32, mode='r+', shape=shape)
                        elif key not in target_buffer.observations:
                            subspace = dummy_env.observation_space.spaces[key]
                            target_buffer.observations[key] = np.zeros((meta["buffer_size"], meta["n_envs"], *subspace.shape), dtype=np.float32)

                    model.rollout_buffer = target_buffer
                    model.rollout_buffer.full = True
                    model.rollout_buffer.pos = meta["buffer_size"]
                    
                    # Log ALL metrics passed from Collector
                    for key, val in meta.items():
                        if any(prefix in key for prefix in ["rollout/", "env/", "success"]):
                            model.logger.record(key, val)
                        elif key not in ["level", "steps", "buffer_size", "n_envs", "observation_space"]:
                            # Log anything else that looks like a metric
                            if isinstance(val, (int, float, np.number)):
                                model.logger.record(f"metrics/{key}", val)

                    # 2. PERFORM PPO UPDATE (+ timings: explains low GPU % = PCIe/Python vs kernel)
                    # Per-minibatch profiling: training.learner_profile in YAML (see parse_learner_profile_option).
                    t_learner_start = time.perf_counter()
                    logger.info("Learner: Starting GPU Update...")
                    try:
                        model.train()
                    finally:
                        # Always clear the ready signal after we bound this buffer, even if
                        # train() raises — otherwise the Collector waits until buffer_consumer_timeout.
                        try:
                            if os.path.exists(signal_path):
                                os.remove(signal_path)
                        except OSError as exc:
                            logger.warning(f"Learner: could not remove buffer signal {signal_path}: {exc}")

                    t_after_train = time.perf_counter()
                    # empty_cache() forces a device sync and allocator churn; it can *lower* throughput
                    # and make GPU util look low while the CPU waits. Opt-in via env only.
                    if os.environ.get("SC2_CUDA_EMPTY_CACHE", "").strip() in ("1", "true", "True"):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Force logger to write to disk
                    model.logger.dump(step=total_steps_trained)

                    # 3. Save updated weights for Collector (Hot Swap)
                    t_before_save = time.perf_counter()
                    tmp_save = self.latest_model_path + ".tmp"
                    model.save(tmp_save)
                    # os.replace overwrites dst; retries handle WinError 32 if collector still
                    # has latest_learner_model.zip open after set_parameters().
                    replace_file_retry(tmp_save, self.latest_model_path)
                    t_after_hot = time.perf_counter()

                    # 4. Save Checkpoint (Permanent Storage)
                    steps_before = total_steps_trained - num_steps
                    did_ckpt = (total_steps_trained // self.save_freq) > (steps_before // self.save_freq)
                    if did_ckpt:
                        ckpt_path = os.path.join(self.model_dir, f"universal_{total_steps_trained}_steps.zip")
                        model.save(ckpt_path)
                        logger.info(f"Saved checkpoint: {ckpt_path}")

                    t_end = time.perf_counter()
                    logger.info(
                        f"Learner timings: train={t_after_train - t_learner_start:.2f}s, "
                        f"hot_swap_save={t_after_hot - t_before_save:.2f}s, "
                        f"total={t_end - t_learner_start:.2f}s"
                    )

                    # Check for completion
                    if total_steps_trained >= training_params["training_steps"]:
                        final_path = os.path.join(self.model_dir, "final_model.zip")
                        model.save(final_path)
                        logger.info(f"Training completed! Saved final model: {final_path}")
                        try:
                            remove_file_retry(self.latest_model_path)
                        except OSError as e:
                            logger.warning(f"Could not remove hot-swap file {self.latest_model_path}: {e}")
                        return

                    logger.info("Learner: Update complete. Weights broadcasted.")
                    found_buffer = True
                    break
            
            if not found_buffer:
                time.sleep(1)


def run_single_async_training_session(
    level_name,
    config,
    model_to_load,
    resume_timesteps_from_save,
    collector_join_timeout=30.0,
):
    """
    One full async train (collector + learner) until training_steps for this level.
    Matches one call to sync UniversalPipelineManager._train_level for a single stage.
    """
    logger.info(f"--- [ASYNC] Target level: {level_name} ---")
    if model_to_load:
        logger.info(f"--- [ASYNC] Loading weights from: {model_to_load} ---")

    force_kill_sc2()
    clear_temp_dir()

    stale_model = os.path.join(TEMP_DIR, "latest_learner_model.zip")
    if os.path.exists(stale_model):
        try:
            remove_file_retry(stale_model)
        except OSError:
            logger.warning(f"Could not remove stale model: {stale_model}")

    stop_event = mp.Event()
    collector = AsyncCollector(
        level_name,
        config,
        stop_event,
        model_path=model_to_load,
        resume_timesteps_from_save=resume_timesteps_from_save,
    )
    collector.start()

    learner = AsyncLearner(
        level_name, config, resume_timesteps_from_save=resume_timesteps_from_save
    )
    try:
        learner.run()
    except KeyboardInterrupt:
        logger.info("Shutdown signaled...")
        raise
    finally:
        if 'learner' in locals() and hasattr(learner, 'close'):
            learner.close()
        stop_event.set()
        collector.terminate()
        collector.join(timeout=collector_join_timeout)
        if collector.is_alive():
            logger.warning("Collector did not exit in time; terminating.")
            try:
                collector.kill()
            except OSError:
                pass
        from src.utils.sc2_cleanup import kill_switch_sc2
        kill_switch_sc2(force=True)

        # Force Python GC and PyTorch CUDA cache clearing
        # Prevents VRAM fragmentation/exhaustion across curriculum levels
        if 'learner' in locals():
            del learner
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    config_dir = "config_universal"
    manager = UniversalPipelineManager(config_dir=config_dir)

    with open(os.path.join(config_dir, "pipeline.yaml"), "r") as f:
        pipeline_config = yaml.safe_load(f)

    from src.utils.config_manager import ConfigManager
    cm = ConfigManager(config_dir)

    # Set to a level name to train only that stage (single run), e.g. "level_move_multi_0"
    override_level = None

    def build_config_for_level(level_name: str):
        cfg = cm.get_level_config(level_name)
        if not cfg:
            return None
        cfg["active_layers"] = cm.get_cumulative_active_layers(level_name)
        cfg["active_tactical_layers"] = cm.get_cumulative_active_tactical_layers(level_name)
        cfg["action_heads"] = cm.get_cumulative_action_heads(level_name)
        cfg["categorical_heads"] = cm.get_cumulative_categorical_heads(level_name)
        cfg["config_dir"] = config_dir
        return cfg

    if override_level:
        level_name = override_level
        status, model_path = manager.get_level_status(level_name)
        if status == "COMPLETED":
            logger.info(f"Override level '{level_name}' already has final_model. Nothing to do.")
            return
        resume_timesteps = status == "CONTINUE_TRAINING"
        if status == "CONTINUE_TRAINING":
            model_to_load = model_path
        elif status == "NEEDS_TRAINING":
            prev_final = None
            for lvl in pipeline_config["levels"]:
                if lvl == level_name:
                    model_to_load = prev_final
                    break
                st, pth = manager.get_level_status(lvl)
                if st == "COMPLETED":
                    prev_final = pth
        else:
            logger.error(f"Unexpected status for override level: {status}")
            sys.exit(1)

        config = build_config_for_level(level_name)
        if not config:
            logger.error(f"Could not load config for {level_name}")
            sys.exit(1)
        try:
            run_single_async_training_session(
                level_name, config, model_to_load, resume_timesteps
            )
        except KeyboardInterrupt:
            pass
        return

    # Full pipeline: same order as UniversalPipelineManager.run() (sync train_universal.py)
    last_final_model_path = None
    for i in range(len(pipeline_config["levels"]) - 1, -1, -1):
        level = pipeline_config["levels"][i]
        status, path = manager.get_level_status(level)
        if status == "COMPLETED":
            last_final_model_path = path
            break

    for level_name in pipeline_config["levels"]:
        status, model_path = manager.get_level_status(level_name)
        if status == "COMPLETED":
            last_final_model_path = model_path
            continue

        config = build_config_for_level(level_name)
        if not config:
            logger.error(f"Could not load config for {level_name}")
            sys.exit(1)

        model_to_load = model_path if status == "CONTINUE_TRAINING" else last_final_model_path
        resume_timesteps = status == "CONTINUE_TRAINING"
        if status == "NEEDS_TRAINING":
            # Only wipe the async subdirectory, not the whole level log directory.
            # _cleanup_logs() deletes logs_universal/<level>/ and everything below it,
            # which includes the async/ subfolder the learner writes to.  When the
            # pipeline is restarted before a first checkpoint exists (NEEDS_TRAINING),
            # that call wipes the new run's event file on every restart — leaving only
            # the 88-byte TensorBoard header and an empty progress.csv.
            async_log_dir = os.path.join(
                "logs_universal", level_name,
                config.get("training", {}).get("tensorboard_subdir", "async"),
            )
            if os.path.exists(async_log_dir):
                try:
                    shutil.rmtree(async_log_dir)
                    logger.info(f"Cleaned up async logs: {async_log_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean async log dir {async_log_dir}: {e}")

        logger.info(
            f"--- [ASYNC] Pipeline stage '{level_name}' "
            f"(status={status}, resume_ckpt_timesteps={resume_timesteps}) ---"
        )
        try:
            run_single_async_training_session(
                level_name, config, model_to_load, resume_timesteps
            )
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
            break

        last_final_model_path = os.path.join(
            manager.model_dir_base, level_name, "final_model.zip"
        )

    logger.info("Universal async pipeline finished (or stopped early).")


if __name__ == "__main__":
    main()

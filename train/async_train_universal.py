import os
import sys
import time
import torch
import numpy as np
import yaml
import json
import gc
from loguru import logger
import multiprocessing as mp
from gymnasium import spaces

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_universal import UniversalPipelineManager, BOT_REGISTRY
from src.sc2_bot_universal import SC2BotUniversal
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.maskable_ppo import MaskablePPO
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.universal_policy import UniversalActorCriticPolicy
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer

import shutil
import psutil

# Standardizing configuration
# Changed to v2 to bypass any zombie AsyncLearner processes from old runs
TEMP_DIR = "C:/temp/sc2_async_buffer_v2"

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

        if self.model_path:
            logger.info(f"Loading weights from: {self.model_path}")
            load_kwargs = {
                "env": vec_env,
                "policy_kwargs": policy_kwargs,
            }
            if AlgoClass == MaskableRecurrentPPO:
                load_kwargs["rnn_type"] = rnn_type
            model = AlgoClass.load(self.model_path, **load_kwargs)
        else:
            algo_kwargs = {
                "policy": PolicyClass,
                "env": vec_env,
                "policy_kwargs": policy_kwargs,
                "n_steps": training_params["n_steps"],
                "batch_size": training_params["batch_size"],
                "gamma": model_params["gamma"]
            }
            if AlgoClass == MaskableRecurrentPPO:
                algo_kwargs["rnn_type"] = rnn_type
            model = AlgoClass(**algo_kwargs)

        if self.model_path and not self.resume_timesteps_from_save:
            model.num_timesteps = 0

        # Diagnostic Log
        obs_shape = model.policy.features_extractor.cnn[0].weight.shape
        logger.info(f"Collector Model initialized with CNN input shape: {obs_shape}")
            
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
            if not current_buffer.wait_for_consumer(timeout=120):
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
                        # Attempt to load parameters
                        logger.info(f"Collector found new weights (mtime={mtime}). Loading...")
                        model.set_parameters(updated_model_path)
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
                    if not hasattr(self, 'last_dones'):
                        self.last_dones = np.zeros(num_envs, dtype=bool)
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
                if not hasattr(self, 'custom_metrics'):
                    self.custom_metrics = {}

                for i, done in enumerate(dones):
                    # Capture metrics from the info dict (e.g., success, etc.)
                    info = infos[i]
                    for key, val in info.items():
                        if isinstance(val, (int, float, np.number)):
                            if key not in self.custom_metrics:
                                self.custom_metrics[key] = []
                            self.custom_metrics[key].append(val)
                            if len(self.custom_metrics[key]) > 100:
                                self.custom_metrics[key].pop(0)

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
        # Ensure log directory exists
        self.log_dir = os.path.join("logs_universal", level_name, "async")
        os.makedirs(self.log_dir, exist_ok=True)
        # Ensure model directory exists
        self.model_dir = os.path.join("models_universal", level_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.save_freq = config["training"].get("save_freq", 50000)

    def run(self):
        logger.info(f"Learner started for level: {self.level_name}")
        training_params = self.config["training"]
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
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "tensorboard_log": self.log_dir,
            "target_kl": 0.015 # SAFETY: Stops epoch early if policy shifts too much
        }
        if AlgoClass == MaskableRecurrentPPO:
            algo_kwargs["rnn_type"] = rnn_type
            
        model = AlgoClass(**algo_kwargs)
        
        # Diagnostic Log
        obs_shape = model.policy.features_extractor.cnn[0].weight.shape
        logger.info(f"Learner Model initialized with CNN input shape: {obs_shape}")
        
        # WAIT FOR COLLECTOR'S INITIAL WEIGHTS
        initial_path = os.path.join(TEMP_DIR, "initial_model.zip")
        logger.info("Learner: Waiting for initial weights from Collector...")
        total_steps_trained = 0
        for _ in range(60): # Wait up to 60 seconds
            if os.path.exists(initial_path):
                try:
                    model.set_parameters(initial_path)

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
        logger.info(f"Learner logging configured to: {self.log_dir}")

        # PRE-INITIALIZE LEARNER BUFFERS
        # We create persistent buffer objects that point to the static directories.
        # This prevents opening/closing file handles during the loop.
        learner_buffers = {}
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
            
            # Pre-load memmaps (Read-Only)
            # These handles will stay open forever.
            # We retry until the file exists because Collector might be slow.
            buf.disk_files = {}
            
            # Placeholder to be filled when files actually exist
            learner_buffers[instance_id] = buf

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
                    if d not in learner_buffers:
                        # Should not happen with static buffering 0/1
                        logger.warning(f"Unknown buffer ID {d}, skipping")
                        continue
                        
                    target_buffer = learner_buffers[d]
                    
                    # LOAD RAM PAYLOAD
                    payload_path = os.path.join(instance_dir, "payload.npz")
                    if os.path.exists(payload_path):
                        try:
                            payload = np.load(payload_path)
                            target_buffer.actions[:] = payload['actions']
                            target_buffer.rewards[:] = payload['rewards']
                            target_buffer.episode_starts[:] = payload['episode_starts']
                            target_buffer.values[:] = payload['values']
                            target_buffer.log_probs[:] = payload['log_probs']
                            target_buffer.advantages[:] = payload['advantages']
                            target_buffer.returns[:] = payload['returns']
                            target_buffer.hidden_states_pi[:] = payload['hidden_states_pi']
                            target_buffer.hidden_states_vf[:] = payload['hidden_states_vf']
                            if 'cell_states_pi' in payload and payload['cell_states_pi'].size > 0:
                                target_buffer.cell_states_pi[:] = payload['cell_states_pi']
                                target_buffer.cell_states_vf[:] = payload['cell_states_vf']
                            if 'action_masks' in payload and payload['action_masks'].size > 0:
                                target_buffer.action_masks[:] = payload['action_masks']
                            
                            # Load RAM-based observations (e.g., 'command')
                            for key in payload.files:
                                if key.startswith("obs_"):
                                    obs_key = key[4:]
                                    if obs_key in target_buffer.observations:
                                        target_buffer.observations[obs_key][:] = payload[key]
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
                                    target_buffer.observations[key] = np.memmap(filename, dtype=np.float32, mode='r', shape=shape)
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

                    # 2. PERFORM PPO UPDATE
                    logger.info("Learner: Starting GPU Update...")
                    model.train()
                    
                    # Force logger to write to disk
                    model.logger.dump(step=total_steps_trained)
                    
                    # 3. Save updated weights for Collector (Hot Swap)
                    tmp_save = self.latest_model_path + ".tmp"
                    model.save(tmp_save)
                    if os.path.exists(self.latest_model_path): os.remove(self.latest_model_path)
                    os.rename(tmp_save, self.latest_model_path)
                    
                    # 4. Save Checkpoint (Permanent Storage)
                    # We check if we crossed a checkpoint boundary
                    # (Simplified: just save if we are "close enough" or exactly on it)
                    # Since steps jump by buffer_size, we use integer division check
                    steps_before = total_steps_trained - num_steps
                    if (total_steps_trained // self.save_freq) > (steps_before // self.save_freq):
                        ckpt_path = os.path.join(self.model_dir, f"universal_{total_steps_trained}_steps.zip")
                        model.save(ckpt_path)
                        logger.info(f"Saved checkpoint: {ckpt_path}")
                        
                    # Check for completion
                    if total_steps_trained >= training_params["training_steps"]:
                        final_path = os.path.join(self.model_dir, "final_model.zip")
                        model.save(final_path)
                        logger.info(f"Training completed! Saved final model: {final_path}")
                        # Clean up temp file
                        if os.path.exists(self.latest_model_path): os.remove(self.latest_model_path)
                        return # Exit learner

                    # 5. Clear signal so Collector can reuse the buffer
                    os.remove(signal_path)
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
            os.remove(stale_model)
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

        model_to_load = model_path if status == "CONTINUE_TRAINING" else last_final_model_path
        resume_timesteps = status == "CONTINUE_TRAINING"
        if status == "NEEDS_TRAINING":
            manager._cleanup_logs(level_name)

        config = build_config_for_level(level_name)
        if not config:
            logger.error(f"Could not load config for {level_name}")
            sys.exit(1)

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

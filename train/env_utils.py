
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers.action_masker import ActionMasker

def get_action_masks(env):
    """Picklable action mask getter."""
    return env.unwrapped.action_masks()

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def vec_env_obs_focus_tags(vec_env, n_envs: int) -> np.ndarray:
    """
    Per-env unit tag for the observation the policy is acting on (see Env.get_obs_focus_unit_tag).
    Used to reset recurrent state when Level3/Level7 cycles to a different unit.
    """
    if not hasattr(vec_env, "env_method"):
        return np.full((n_envs,), -1, dtype=np.int64)
    tags = vec_env.env_method("get_obs_focus_unit_tag")
    arr = np.asarray(tags, dtype=np.int64).reshape((n_envs,))
    return arr

def create_dummy_env_from_config(config: dict):
    """
    Creates a mock Gymnasium environment with only spaces defined.
    Used by the Async Learner to initialize models without launching SC2.
    """
    env_params = config.get("env", {})
    grid_size = env_params.get("grid_size", 128)
    spatial_res = 32
    
    # Matches Universal Architecture spaces
    # Keep global and tactical channels independent (they can differ).
    active_layers = config.get("active_layers") or config.get("env", {}).get("active_layers", [])
    active_tactical_layers = config.get("active_tactical_layers") or config.get("env", {}).get("active_tactical_layers", [])
    num_layers = len(active_layers) if active_layers else 30
    num_tactical_layers = len(active_tactical_layers) if active_tactical_layers else num_layers
    print(
        "DEBUG: create_dummy_env_from_config -> "
        f"map_layers={num_layers}, tactical_layers={num_tactical_layers}, "
        f"active_layers={active_layers}, active_tactical_layers={active_tactical_layers}"
    )
        
    obs_dict = {
        "map": spaces.Box(low=0.0, high=1.0, shape=(num_layers, 128, 128), dtype=np.float32),
        "tactical": spaces.Box(low=0.0, high=1.0, shape=(num_tactical_layers, 32, 32), dtype=np.float32),
        "command": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
    }

    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict(obs_dict)
            
            # Use accumulated action_heads from top-level config first,
            # then nested env config as fallback.
            self.action_heads = config.get("action_heads") or config.get("env", {}).get("action_heads")
            if not self.action_heads:
                self.action_heads = ["move_global", "move_local"]
                
            self.action_space = spaces.MultiDiscrete([len(self.action_heads), spatial_res * spatial_res])

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return ActionMasker(DummyEnv(), lambda e: [np.ones(d, dtype=bool) for d in e.action_space.nvec])

class EnvFactory:

    """
    Picklable factory for environment creation, necessary for SubprocVecEnv on Windows.
    """
    def __init__(self, env_class, level_name, visualize, grid_size, rank, num_envs, 
                 sc2_process_cls, client_cls, training_conf, max_steps, initial_steps=0):
        self.env_class = env_class
        self.level_name = level_name
        self.visualize = visualize
        self.grid_size = grid_size
        self.rank = rank
        self.num_envs = num_envs
        self.sc2_process_cls = sc2_process_cls
        self.client_cls = client_cls
        self.training_conf = training_conf
        self.max_steps = max_steps
        self.initial_steps = initial_steps

    def __call__(self):
        env = self.env_class(
            level_name=self.level_name,
            visualize=self.visualize,
            grid_size=self.grid_size,
            worker_index=self.rank,
            num_workers=self.num_envs,
            sc2_process_cls=self.sc2_process_cls,
            client_cls=self.client_cls,
            training_conf=self.training_conf,
            max_steps=self.max_steps,
            initial_steps=self.initial_steps
        )
        # Wrap with Monitor first, then ActionMasker
        env = Monitor(env)
        env = ActionMasker(env, get_action_masks)
        return env

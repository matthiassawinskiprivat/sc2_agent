"""Obs focal unit tag for recurrent training (Level3/7 cycling)."""
import numpy as np
from unittest.mock import MagicMock

from train.env_utils import vec_env_obs_focus_tags
from train.envs.level_3_env import Level3Env


def test_vec_env_obs_focus_tags_from_env_method():
    vec = MagicMock()
    vec.env_method = MagicMock(return_value=[7, 8, 9])
    arr = vec_env_obs_focus_tags(vec, 3)
    assert arr.dtype == np.int64
    assert arr.tolist() == [7, 8, 9]


def test_sync_vec_env_env_method_unwraps_action_masker():
    """num_envs=1 uses SyncVecEnv; methods on the base env must work through ActionMasker."""
    from gymnasium import spaces
    from gymnasium.core import Env
    from sb3_contrib.common.wrappers.action_masker import ActionMasker
    from train.sync_vec_env import SyncVecEnv
    from train.env_utils import get_action_masks, vec_env_obs_focus_tags, vec_env_requires_policy_action

    class BaseEnv(Env):
        metadata = {}

        def __init__(self):
            self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Discrete(2)
            self._tag = 42

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

        def action_masks(self):
            return np.array([True, False], dtype=bool)

        def get_obs_focus_unit_tag(self):
            return self._tag

        def requires_policy_action(self):
            return True

    def make_env():
        return ActionMasker(BaseEnv(), get_action_masks)

    vec = SyncVecEnv([make_env])
    assert vec_env_obs_focus_tags(vec, 1).tolist() == [42]
    assert vec_env_requires_policy_action(vec, 1).tolist() == [True]
    vec.close()


def test_sync_vec_env_dict_obs_is_vectorized_for_sb3():
    """num_envs=1 SyncVecEnv must stack Dict obs per-key, not np.stack the dict list."""
    from gymnasium import spaces
    from gymnasium.core import Env
    from stable_baselines3.common.utils import is_vectorized_observation

    from train.sync_vec_env import SyncVecEnv

    class DictObsEnv(Env):
        metadata = {}

        def __init__(self):
            self.observation_space = spaces.Dict(
                {
                    "map": spaces.Box(low=0, high=1, shape=(2, 4, 4), dtype=np.float32),
                    "command": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
                }
            )
            self.action_space = spaces.MultiDiscrete([2, 4])

        def reset(self, *, seed=None, options=None):
            return self._sample_obs(), {}

        def step(self, action):
            return self._sample_obs(), 0.0, False, False, {}

        def _sample_obs(self):
            return {
                "map": self.observation_space["map"].sample(),
                "command": self.observation_space["command"].sample(),
            }

    vec = SyncVecEnv([DictObsEnv])
    obs = vec.reset()
    assert is_vectorized_observation(obs, vec.observation_space)
    assert obs["map"].shape == (1, 2, 4, 4)
    assert obs["command"].shape == (1, 3)

    vec.step_async(np.array([[0, 1]]))
    obs, _, _, _ = vec.step_wait()
    assert is_vectorized_observation(obs, vec.observation_space)
    assert obs["map"].shape == (1, 2, 4, 4)
    vec.close()


def test_level3_get_obs_focus_unit_tag_minimal():
    env = object.__new__(Level3Env)
    env.active_unit_tags = [100, 200, 300]
    env.current_unit_index = 2
    assert Level3Env.get_obs_focus_unit_tag(env) == 300


def test_level3_get_obs_focus_empty_tags():
    env = object.__new__(Level3Env)
    env.active_unit_tags = []
    env.current_unit_index = 0
    assert Level3Env.get_obs_focus_unit_tag(env) == -1

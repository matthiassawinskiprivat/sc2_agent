
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Type

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info


class SyncVecEnv(VecEnv):
    """
    A synchronous, custom vectorized environment.
    """
    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)

        self.keys, shapes, dtypes = obs_space_info(env.observation_space)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos: List[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for i, (env, action) in enumerate(zip(self.envs, self.actions)):
            logger.debug(f"SyncVecEnv: Stepping environment {i}...")
            ob, self.buf_rews[i], terminated, truncated, self.buf_infos[i] = env.step(action)
            self.buf_dones[i] = terminated or truncated
            self.buf_infos[i]["TimeLimit.truncated"] = truncated and not terminated
            if self.buf_dones[i]:
                logger.debug(f"SyncVecEnv: Environment {i} done. Resetting...")
                self.buf_infos[i]["terminal_observation"] = ob
                ob, _ = env.reset()
            self._save_obs(i, ob)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset(self):
        for i, env in enumerate(self.envs):
            logger.debug(f"SyncVecEnv: Resetting environment {i}...")
            ob, _ = env.reset()
            self._save_obs(i, ob)
        return self._obs_from_buf()

    def close(self) -> None:
        for i, env in enumerate(self.envs):
            logger.debug(f"SyncVecEnv: Closing environment {i}...")
            env.close()

    def _save_obs(self, env_idx: int, obs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        """Return attribute from vectorized environment."""
        target_envs = self._get_target_envs(indices)
        out = []
        for env_i in target_envs:
            if isinstance(env_i, gym.Wrapper):
                out.append(env_i.get_wrapper_attr(attr_name))
            else:
                out.append(getattr(env_i, attr_name))
        return out

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        """Set attribute inside vectorized environments."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def _call_env_method(self, env_i: gym.Env, method_name: str, method_args, method_kwargs):
        """Resolve methods on wrapped envs (Monitor/ActionMasker), matching SubprocVecEnv workers."""
        if hasattr(env_i, "get_wrapper_attr"):
            method = env_i.get_wrapper_attr(method_name)
            return method(*method_args, **method_kwargs)
        return getattr(env_i, method_name)(*method_args, **method_kwargs)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            self._call_env_method(env_i, method_name, method_args, method_kwargs)
            for env_i in target_envs
        ]

    def _get_target_envs(self, indices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: Any = None) -> List[bool]:
        target_envs = self._get_target_envs(indices)
        return [isinstance(env_i, wrapper_class) for env_i in target_envs]

    def get_action_masks(self) -> np.ndarray:
        return np.stack(self.env_method("action_masks")).astype(bool)

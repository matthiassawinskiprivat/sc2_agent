
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np
import gymnasium as gym
from loguru import logger

class SyncVecEnv(VecEnv):
    """
    A synchronous, custom vectorized environment.
    """
    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        super().__init__(len(env_fns), self.envs[0].observation_space, self.envs[0].action_space)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, self.actions)):
            logger.debug(f"SyncVecEnv: Stepping environment {i}...")
            ob, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                logger.debug(f"SyncVecEnv: Environment {i} done. Resetting...")
                ob, _ = env.reset()
            obs.append(ob)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
        return np.stack(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        obs = []
        for i, env in enumerate(self.envs):
            logger.debug(f"SyncVecEnv: Resetting environment {i}...")
            ob, _ = env.reset()
            obs.append(ob)
        return np.stack(obs)

    def close(self) -> None:
        for i, env in enumerate(self.envs):
            logger.debug(f"SyncVecEnv: Closing environment {i}...")
            env.close()

    def get_attr(self, attr_name: str, indices = None) -> List[Any]:
        """Return attribute from vectorized environment."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices = None) -> None:
        """Set attribute inside vectorized environments."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: Any = None) -> List[bool]:
        # Implementation for env_is_wrapped, assuming it checks if the environment is wrapped
        # with a certain class. This is a basic implementation.
        target_envs = self._get_target_envs(indices)
        return [isinstance(env_i, wrapper_class) for env_i in target_envs]
    
    def get_action_masks(self) -> np.ndarray:
        return np.stack([env.action_masks() for env in self.envs])

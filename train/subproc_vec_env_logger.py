import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

class SubprocVecEnvLogger(BaseCallback):
    """
    A custom callback to log episode statistics from a SubprocVecEnv.
    This callback accesses the `get_episode_rewards` and `get_episode_lengths` methods
    of the Monitor wrapper in each sub-environment and logs their mean.
    """
    def __init__(self, verbose: int = 0):
        super(SubprocVecEnvLogger, self).__init__(verbose)
        self.last_log_time = time.time()

    def _on_step(self) -> bool:
        if not isinstance(self.training_env, SubprocVecEnv):
            if self.verbose > 0:
                print("Warning: SubprocVecEnvLogger is only designed to work with SubprocVecEnv.")
            return True

        # Only log at the end of a rollout, if enough time has passed.
        # This prevents excessive logging and ensures metrics are updated.
        if self.n_calls % self.model.n_steps != 0 or (time.time() - self.last_log_time < 1.0):
            return True
        
        self.last_log_time = time.time()

        # Retrieve episode rewards and lengths from all sub-environments
        all_rewards = self.training_env.env_method('get_episode_rewards')
        all_lengths = self.training_env.env_method('get_episode_lengths')

        # Flatten the lists of lists
        flat_rewards = [item for sublist in all_rewards for item in sublist]
        flat_lengths = [item for sublist in all_lengths for item in sublist]

        if not flat_rewards:
            return True

        ep_len_mean = np.mean(flat_lengths)
        ep_rew_mean = np.mean(flat_rewards)

        self.logger.record('rollout/ep_len_mean', ep_len_mean)
        self.logger.record('rollout/ep_rew_mean', ep_rew_mean)

        return True

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers.action_masker import ActionMasker

def get_action_masks(env):
    """Picklable action mask getter."""
    return env.unwrapped.action_masks()

class EnvFactory:
    """
    Picklable factory for environment creation, necessary for SubprocVecEnv on Windows.
    """
    def __init__(self, env_class, level_name, visualize, grid_size, rank, num_envs, 
                 sc2_process_cls, client_cls, training_conf, max_steps):
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
            max_steps=self.max_steps
        )
        # Wrap with Monitor first, then ActionMasker
        env = Monitor(env)
        env = ActionMasker(env, get_action_masks)
        return env

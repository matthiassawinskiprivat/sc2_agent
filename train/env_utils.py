
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers.action_masker import ActionMasker

def get_action_masks(env):
    """Picklable action mask getter."""
    return env.unwrapped.action_masks()

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM


def is_squad_pointer_action_space(action_space) -> bool:
    """True when vec-env uses squad pointer MultiDiscrete layout.

    Layout is ``[2]*N_max + [command, target_slot, spatial]`` (see
    ``squad_action_space_nvec`` / ``SQUAD_ACTION_DIM``). The older who→cmd→where
    layout was ``N_max + 2``; detecting that length here would mis-route the
    async collector onto the universal per-unit batch helper and feed ``tactical``
    obs into a squad policy (``KeyError: 'tactical'``).
    """
    from src.agent.squad_constants import MAX_SQUAD_UNITS, SQUAD_ACTION_DIM

    nvec = getattr(action_space, "nvec", None)
    if nvec is None or len(nvec) != SQUAD_ACTION_DIM:
        return False
    return all(int(n) == 2 for n in nvec[:MAX_SQUAD_UNITS])


def resolve_level_agent_architecture(config: dict, level_name: str | None = None) -> str:
    arch = config.get("agent_architecture")
    if arch:
        return str(arch)
    env_arch = config.get("env", {}).get("agent_architecture")
    if env_arch:
        return str(env_arch)
    level = level_name or config.get("level_name")
    if level:
        from src.utils.config_manager import ConfigManager

        cm = ConfigManager(config.get("config_dir", "config_universal"))
        return cm.get_agent_architecture(level)
    return "universal"


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


def vec_env_requires_policy_action(vec_env, n_envs: int) -> np.ndarray:
    """
    Per-env flag: True when the collector should run policy forward and buffer the transition.
    CTF opponent squads return False (env runs internal opponent batch).
    """
    if not hasattr(vec_env, "env_method"):
        return np.ones((n_envs,), dtype=bool)
    try:
        flags = vec_env.env_method("requires_policy_action")
    except Exception:
        return np.ones((n_envs,), dtype=bool)
    return np.asarray(flags, dtype=bool).reshape((n_envs,))

def create_dummy_env_from_config(config: dict, level_name: str | None = None):
    """
    Creates a mock Gymnasium environment with only spaces defined.
    Used by the Async Learner to initialize models without launching SC2.
    """
    env_params = config.get("env", {})
    grid_size = env_params.get("grid_size", 128)
    spatial_res = 32

    config_dir = config.get("config_dir", "config_universal")
    level = level_name or config.get("level_name")
    agent_architecture = resolve_level_agent_architecture(config, level)
    if level:
        from src.utils.config_manager import ConfigManager

        cm = ConfigManager(config_dir)
        active_layers = cm.get_cumulative_active_layers(level)
        active_tactical_layers = cm.get_cumulative_active_tactical_layers(level)
        active_unit_scalars = cm.get_cumulative_active_unit_scalars(level)
        action_heads = cm.get_cumulative_action_heads(level)
    else:
        active_layers = config.get("active_layers") or config.get("env", {}).get("active_layers", [])
        active_tactical_layers = config.get("active_tactical_layers") or config.get(
            "env", {}
        ).get("active_tactical_layers", [])
        active_unit_scalars = config.get("active_unit_scalars") or config.get(
            "env", {}
        ).get("active_unit_scalars", [])
        action_heads = config.get("action_heads") or config.get("env", {}).get("action_heads")
    
    num_layers = len(active_layers) if active_layers else 30
    num_tactical_layers = len(active_tactical_layers) if active_tactical_layers else num_layers
    print(
        "DEBUG: create_dummy_env_from_config -> "
        f"arch={agent_architecture}, map_layers={num_layers}, "
        f"tactical_layers={num_tactical_layers}"
    )

    obs_dict = {
        "map": spaces.Box(low=0.0, high=1.0, shape=(num_layers, 128, 128), dtype=np.float32),
        "command": spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32),
    }

    if agent_architecture == "squad":
        from src.agent.squad_constants import (
            MAX_SQUAD_ENEMIES,
            MAX_SQUAD_UNITS,
            SQUAD_COMMAND_DIM,
            SQUAD_COMMAND_MODE_DIM,
            SQUAD_COMMAND_MODE_MAX,
            SQUAD_RACE_DIM,
            SQUAD_UPGRADES_DIM,
            include_race_enabled,
            include_upgrades_enabled,
            squad_action_space_nvec,
        )
        from src.agent.squad_target_masks import SQUAD_TARGET_MASK_KEYS
        from src.agent.unit_scalar_constants import UnitScalarConstants

        resolved_scalars = UnitScalarConstants.resolve_active_features(active_unit_scalars)
        n_unit_scalars = len(resolved_scalars)

        # Prefer the level YAML ``env`` block when ``level_name`` is set so sparse
        # configs (learner startup) still honor ``include_race: false`` on level 01.
        race_conf = config
        if level:
            try:
                level_cfg = cm.get_level_config(level)
                race_conf = {**level_cfg, **config}
            except Exception:
                race_conf = config

        obs_dict["command"] = spaces.Box(
            low=0.0, high=1.0, shape=(SQUAD_COMMAND_DIM,), dtype=np.float32
        )
        obs_dict["anchor_tactical"] = spaces.Box(
            low=0.0, high=1.0, shape=(num_tactical_layers, 32, 32), dtype=np.float32
        )
        obs_dict["unit_scalars"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_UNITS, n_unit_scalars), dtype=np.float32
        )
        obs_dict["unit_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_UNITS,), dtype=np.float32
        )
        # Must match CaptureTheFlagSingleEnv squad spaces — omitting enemy_* drops
        # SquadNetwork.enemy_encoder on the learner, so latest_learner_model.zip is
        # missing those keys and collector hot-reload fails (strict set_parameters).
        obs_dict["enemy_scalars"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_ENEMIES, n_unit_scalars), dtype=np.float32
        )
        obs_dict["enemy_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_ENEMIES,), dtype=np.float32
        )
        obs_dict["tactical_enemy_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_ENEMIES,), dtype=np.float32
        )
        obs_dict["engage_enemy_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_ENEMIES,), dtype=np.float32
        )
        obs_dict["snipe_target_mask"] = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_SQUAD_ENEMIES,), dtype=np.float32
        )
        obs_dict["command_modes"] = spaces.Box(
            low=0.0,
            high=SQUAD_COMMAND_MODE_MAX,
            shape=(SQUAD_COMMAND_MODE_DIM,),
            dtype=np.float32,
        )
        for key in SQUAD_TARGET_MASK_KEYS:
            obs_dict[key] = spaces.Box(
                low=0.0, high=1.0, shape=(MAX_SQUAD_UNITS,), dtype=np.float32
            )
        if include_race_enabled(race_conf):
            obs_dict["race"] = spaces.Box(
                low=0.0, high=1.0, shape=(SQUAD_RACE_DIM,), dtype=np.float32
            )
        if include_upgrades_enabled(race_conf):
            obs_dict["upgrades"] = spaces.Box(
                low=0.0, high=1.0, shape=(SQUAD_UPGRADES_DIM,), dtype=np.float32
            )
        action_space = spaces.MultiDiscrete(squad_action_space_nvec())
    else:
        obs_dict["tactical"] = spaces.Box(
            low=0.0, high=1.0, shape=(num_tactical_layers, 32, 32), dtype=np.float32
        )
        action_space = spaces.MultiDiscrete(
            [FIXED_UNIVERSAL_COMMAND_DIM, spatial_res * spatial_res]
        )

    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict(obs_dict)
            self.agent_architecture = agent_architecture
            
            self.action_heads = action_heads
            if not self.action_heads:
                self.action_heads = ["move_global", "move_local"]

            self.action_space = action_space

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    def _dummy_action_masks(env):
        inner = env.unwrapped
        if inner.agent_architecture == "squad":
            from src.agent.squad_constants import MAX_SQUAD_UNITS

            flat_len = int(np.sum(inner.action_space.nvec))
            mask = np.ones(flat_len, dtype=bool)
            cmd_start = MAX_SQUAD_UNITS * 2
            n_def = min(FIXED_UNIVERSAL_COMMAND_DIM, len(inner.action_heads))
            mask[cmd_start : cmd_start + FIXED_UNIVERSAL_COMMAND_DIM] = False
            mask[cmd_start : cmd_start + n_def] = True
            return mask
        cmd = np.zeros(FIXED_UNIVERSAL_COMMAND_DIM, dtype=bool)
        n_def = len(inner.action_heads)
        cmd[:n_def] = True
        spatial = np.ones(spatial_res * spatial_res, dtype=bool)
        return np.concatenate([cmd, spatial])

    return ActionMasker(DummyEnv(), _dummy_action_masks)

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

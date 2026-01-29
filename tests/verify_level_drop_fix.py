import os
from train.envs.level_drop_env import LevelDropEnv
from src.utils.config_manager import ConfigManager

def verify_level_drop_resolution():
    # Mock some expected context for the env
    level_name = "level_drop_0"
    
    # We need to ensure we are in a state where ConfigManager can find the configs
    config_dir = "config"
    cm = ConfigManager(config_dir)
    
    active_layers = cm.get_cumulative_active_layers(level_name)
    action_heads = cm.get_cumulative_action_heads(level_name)
    
    print(f"Resolved Layers ({len(active_layers)}): {active_layers}")
    print(f"Resolved Action Heads ({len(action_heads)}): {action_heads}")
    
    # Instantiate Env
    env = LevelDropEnv(
        grid_size=128,
        level_name=level_name
    )
    
    print(f"Env Action Space: {env.action_space.n}")
    print(f"Env Observation Space: {env.observation_space.shape}")
    
    expected_actions = 4 * 128 * 128
    if env.action_space.n == expected_actions:
        print("SUCCESS: Action space matches expectation (65536)")
    else:
        print(f"FAILURE: Action space {env.action_space.n} != {expected_actions}")

    if env.observation_space.shape[0] == 35:
        print("SUCCESS: Observation channels match expectation (35)")
    else:
        print(f"FAILURE: Observation channels {env.observation_space.shape[0]} != 35")

if __name__ == "__main__":
    verify_level_drop_resolution()

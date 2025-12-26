
import yaml
import os
import sys
from train.envs.level_0_env import Level0Env

def reproduce_integration():
    print("Loading config/level_0.yaml...")
    with open("config/level_0.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    training_params = config.get("training", {})
    print(f"Training params from YAML: {training_params}")
    
    if not training_params.get("render_beacon"):
        print("FAILURE: render_beacon is missing or False in YAML!")
        return

    print("Initializing Level0Env with params...")
    try:
        env = Level0Env(training_conf=training_params)
    except FileNotFoundError:
        print("Warning: Map file not found, skipping map check, but checking config retention.")
        # We can mock map discovery if needed, but Level0Env might fail init
        # Let's bypass map loading if possible or just assume it fails later
        # Actually Level0Env fails in __init__ if no maps found.
        # But we are in the project root, maps should be there.
    
    print(f"Env.training_conf: {env.training_conf}")
    
    if env.training_conf.get("render_beacon") is not True:
        print("FAILURE: Env.training_conf['render_beacon'] is NOT True!")
    else:
        print("SUCCESS: Env initialized with correct config.")

    # Now let's check SC2Bot creation logic (without launching SC2)
    # We can manually trigger what _setup_game_session does regarding bot creation
    from src.sc2_bot import SC2Bot
    
    print("Creating SC2Bot manually with env.training_conf...")
    bot = SC2Bot(conn=None, training_conf=env.training_conf)
    
    print(f"Bot.training_conf: {bot.training_conf}")
    
    if bot.training_conf.get("render_beacon") is not True:
        print("FAILURE: Bot.training_conf['render_beacon'] is NOT True!")
        return

    # Now check HierarchicalAgentSystem creation logic
    # Mimic on_step(0)
    from src.hierarchical_agent import HierarchicalAgentSystem
    
    print("Creating HierarchicalAgentSystem...")
    agent_system = HierarchicalAgentSystem(
        map_size=(128, 128),
        grid_size=128,
        training_conf=bot.training_conf
    )
    
    # Access the transformer
    transformer = agent_system.shared_transformer.transformer
    print(f"Transformer.training_conf: {transformer.training_conf}")
    
    if transformer.training_conf.get("render_beacon") is not True:
         print("FAILURE: Transformer.training_conf['render_beacon'] is NOT True!")
    else:
         print("SUCCESS: Transformer received correct config.")

if __name__ == "__main__":
    reproduce_integration()

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import builtins
from unittest.mock import patch, mock_open
from train.train_pipeline import PipelineManager
from loguru import logger

def test_pipeline_init_level_drop():
    # Use the drop-specific pipeline config
    manager = PipelineManager(config_dir="config")
    manager.pipeline_config = {"levels": ["level_drop_0"]}
    
    # Path to the CTF model we want to resume from
    model_to_resume = "models/capture_the_flag_zerg/final_model.zip"
    
    if not os.path.exists(model_to_resume):
        print(f"ERROR: Base model {model_to_resume} not found for integration test.")
        return

    # Update level drop config for a very short run
    from src.utils.config_manager import ConfigManager
    cm = ConfigManager("config")
    level_conf = cm.get_level_config("level_drop_0")
    level_conf["training"]["training_steps"] = 100
    level_conf["training"]["num_envs"] = 1
    
    # Dump modified config to string
    modified_yaml_str = yaml.dump(level_conf)
    
    # Save original open to call it for non-matching files
    original_open = builtins.open

    def mocked_open(file, *args, **kwargs):
        # Check if file matches the config we want to intercept
        # We handle string paths only
        if isinstance(file, (str, bytes, os.PathLike)):
            s_file = str(file).replace("\\", "/")
            if "config/level_drop_0.yaml" in s_file:
                # Return a mock that simulates reading the modified yaml
                print(f"[TEST] Intercepting read for {file}, supplying mocked config with 100 steps.")
                return mock_open(read_data=modified_yaml_str)(file, *args, **kwargs)
        
        # Fallback to real open
        return original_open(file, *args, **kwargs)
    
    try:
        # Run training for the level
        # We patch 'builtins.open' to inject our modified config
        with patch('builtins.open', side_effect=mocked_open):
            # We call it with "level_drop_0", which will try to open "config/level_drop_0.yaml"
            # Our mock will intercept this specific call.
            manager._train_level("level_drop_0", model_path_to_load=model_to_resume)
            
        print("SUCCESS: Pipeline initialized and ran successfully.")
    except Exception as e:
        print(f"FAILURE: Pipeline failed during LevelDrop initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure logs don't spam too much
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    test_pipeline_init_level_drop()
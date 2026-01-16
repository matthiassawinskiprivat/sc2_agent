import os
import sys
import torch
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
    
    # Write back temp config
    import yaml
    with open("config/level_drop_0_tmp.yaml", "w") as f:
        yaml.dump(level_conf, f)
    
    try:
        # Run training for the level
        # We use a custom level name to avoid polluting existing models
        # but PipelineManager._train_level expects config to be in config/level_name.yaml
        
        # Let's just run it for level_drop_0 but ensure it uses the CTF model
        manager._train_level("level_drop_0", model_path_to_load=model_to_resume)
        print("SUCCESS: Pipeline initialized and ran successfully.")
    except Exception as e:
        print(f"FAILURE: Pipeline failed during LevelDrop initialization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists("config/level_drop_0_tmp.yaml"):
            os.remove("config/level_drop_0_tmp.yaml")

if __name__ == "__main__":
    # Ensure logs don't spam too much
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    test_pipeline_init_level_drop()

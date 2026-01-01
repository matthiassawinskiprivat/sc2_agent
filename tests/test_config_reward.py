from src.utils.config_manager import ConfigManager
import os

def test_config_loading():
    cm = ConfigManager("config")
    
    # Test Level 2 specifically
    l2_conf = cm.get_level_config("level_2")
    
    assert l2_conf['training']['n_steps'] == 2048
    assert l2_conf['model']['gamma'] == 0.995
    assert l2_conf['env']['rewards']['success'] == 100.0
    assert l2_conf['env']['rewards']['step_penalty'] == -0.075
    
    assert l2_conf['env']['rewards']['distance_multiplier'] == 0.5
    
    print("Config loading test passed!")

if __name__ == "__main__":
    test_config_loading()

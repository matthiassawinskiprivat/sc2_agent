import pytest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.envs.multi_client_env import MultiClientEnv
from src.utils.config_manager import ConfigManager

def test_multi_client_env_has_config_manager():
    # We initialize MultiClientEnv with minimum required params
    # We don't need a real SC2 process for this attribute check
    env = MultiClientEnv(map_name="Simple64")
    
    assert hasattr(env, "config_manager")
    assert isinstance(env.config_manager, ConfigManager)
    assert env.config_manager.config_dir == "config"

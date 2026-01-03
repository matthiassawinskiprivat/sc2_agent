
import pytest
import os
import yaml
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from train.envs.capture_the_flag_env import CaptureTheFlagEnv

def test_ctf_single_env_config_loading():
    # Simulate how PipelineManager creates the config
    training_params = {
        "num_envs": 1,
        "n_steps": 2048,
        "batch_size": 512,
        "learning_rate": 0.0003
    }
    env_params = {
        "mineral_budget": 1000,
        "vespene_budget": 250,
        "unit_pool": ["ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT"],
        "grid_size": 128
    }
    
    combined_conf = training_params.copy()
    combined_conf.update(env_params)
    
    # Instantiate environment as PipelineManager does
    env = CaptureTheFlagSingleEnv(
        training_conf=combined_conf,
        grid_size=env_params["grid_size"]
    )
    
    # Check if config was loaded correctly
    # If the bug exists, these will fail
    assert env.mineral_budget == 1000, f"Expected 1000 minerals, got {env.mineral_budget}"
    assert env.vespene_budget == 250, f"Expected 250 vespene, got {env.vespene_budget}"
    assert env.unit_pool == ["ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT"], f"Expected full unit pool, got {env.unit_pool}"

def test_ctf_multi_env_config_loading():
    # Simulate how PipelineManager creates the config
    training_params = {
        "num_envs": 1,
        "n_steps": 2048,
        "batch_size": 512,
        "learning_rate": 0.0003
    }
    env_params = {
        "mineral_budget": 1000,
        "vespene_budget": 250,
        "unit_pool": ["ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT"],
        "grid_size": 128
    }
    
    combined_conf = training_params.copy()
    combined_conf.update(env_params)
    
    # Instantiate environment as PipelineManager does
    env = CaptureTheFlagEnv(
        training_conf=combined_conf,
        grid_size=env_params["grid_size"]
    )
    
    # Check if config was loaded correctly
    assert env.minerals == 1000, f"Expected 1000 minerals, got {env.minerals}"
    assert env.vespene == 250, f"Expected 250 vespene, got {env.vespene}"
    assert env.unit_pool == ["ZERGLING", "ROACH", "MARINE", "REAPER", "ZEALOT"], f"Expected full unit pool, got {env.unit_pool}"

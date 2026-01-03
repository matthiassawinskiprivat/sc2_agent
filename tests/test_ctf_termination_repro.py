
import pytest
import numpy as np
from loguru import logger
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

def test_ctf_timeout_logic():
    """Verify that the environment respects max_steps and truncates correctly."""
    # Set a small max_steps for testing
    MAX_STEPS = 20
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=MAX_STEPS)
    
    try:
        obs, info = env.reset()
        assert env.steps_count == 0
        
        # Step up to MAX_STEPS - 1
        for i in range(MAX_STEPS - 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Should NOT end yet
            if terminated or truncated:
                logger.error(f"Premature termination at step {env.steps_count}. Terminated: {terminated}, Truncated: {truncated}")
            
            assert not terminated, f"Game terminated prematurely at step {env.steps_count}"
            assert not truncated, f"Game truncated prematurely at step {env.steps_count}"
            assert env.steps_count == i + 1

        # Final step to trigger timeout
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.steps_count == MAX_STEPS
        assert truncated, "Game did not truncate at max_steps"
        assert not terminated, "Game should be truncated, not terminated (unless pylon died by chance)"
        
    finally:
        env.close()

def test_ctf_premature_termination():
    """Verify that the environment does NOT terminate early with high max_steps."""
    # Set a higher max_steps
    MAX_STEPS = 50
    env = CaptureTheFlagSingleEnv(visualize=False, max_steps=MAX_STEPS)
    
    try:
        obs, info = env.reset()
        
        # Step through
        for _ in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # We expect NO termination unless a Pylon dies (unlikely in 50 random steps but possible)
            # However, we are specifically checking for "random" premature ends not caused by game mechanics
            # So we check if terminated is True but NO pylon is dead (we'd need access to bot state for that)
            
            if terminated:
                 # Check if a pylon is actually dead
                 p1_pylon_count = env.bot.structures(env.bot.unit_type_id.PYLON).amount
                 p2_pylon_count = env.bot.enemy_structures(env.bot.unit_type_id.PYLON).amount
                 if p1_pylon_count > 0 and p2_pylon_count > 0:
                     pytest.fail(f"Terminated is True but both pylons are alive! Step: {env.steps_count}")
            
            if truncated and env.steps_count < MAX_STEPS:
                pytest.fail(f"Truncated is True before max_steps! Step: {env.steps_count}")

    finally:
        env.close()

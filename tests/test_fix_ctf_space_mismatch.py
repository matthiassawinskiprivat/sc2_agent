
import pytest
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv

def test_ctf_observation_space_shape():
    """
    FAILING TEST: CTF should have at least 9 layers to match Level 7.
    Current implementation has 7.
    """
    env = CaptureTheFlagEnv()
    # The requirement is to have ALL previous layers + any new ones.
    # Level 7 had 9. CTF should have at least 9.
    assert env.observation_space.shape[0] >= 9, f"Expected at least 9 observation layers, got {env.observation_space.shape[0]}"

def test_ctf_action_space_size():
    """
    FAILING TEST: CTF should have 32768 actions (2 * 128 * 128) to match Level 7 dual head.
    Current implementation has 16384.
    """
    env = CaptureTheFlagEnv()
    # 2 grids of 128x128 = 32768
    assert env.action_space.n == 32768, f"Expected 32768 actions, got {env.action_space.n}"

def test_ctf_action_mask_shape():
    """
    FAILING TEST: Action mask must match the new action space size.
    """
    env = CaptureTheFlagEnv()
    mask = env.action_masks()
    assert mask.shape[0] == 32768, f"Expected mask size 32768, got {mask.shape[0]}"

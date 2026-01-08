
import pytest
import numpy as np
from loguru import logger
from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
from sc2.ids.unit_typeid import UnitTypeId

def test_ctf_pylon_counts_and_positions():
    """Verify that each player has exactly one pylon and positions are randomized."""
    env = CaptureTheFlagSingleEnv(visualize=False)
    
    try:
        # First reset
        obs, info = env.reset()
        
        p1_pylons = env.bot.structures(UnitTypeId.PYLON)
        p2_pylons = env.bot.enemy_structures(UnitTypeId.PYLON)
        
        assert len(p1_pylons) == 1, f"Player 1 should have 1 pylon, found {len(p1_pylons)}"
        assert len(p2_pylons) == 1, f"Player 2 should have 1 pylon, found {len(p2_pylons)}"
        
        pos1_a = p1_pylons.first.position
        pos2_a = p2_pylons.first.position
        
        dist_a = pos1_a.distance_to(pos2_a)
        flag_dist_min = env.training_conf.get("env", {}).get("flag_distance_min", 20.0)
        assert dist_a >= flag_dist_min, f"Distance {dist_a} is less than min {flag_dist_min}"
        
        # Second reset
        obs, info = env.reset()
        
        p1_pylons = env.bot.structures(UnitTypeId.PYLON)
        p2_pylons = env.bot.enemy_structures(UnitTypeId.PYLON)
        
        assert len(p1_pylons) == 1
        assert len(p2_pylons) == 1
        
        pos1_b = p1_pylons.first.position
        pos2_b = p2_pylons.first.position
        
        # Check for randomization (extremely unlikely to be identical)
        assert pos1_a != pos1_b, "Player 1 pylon position did not change after reset"
        assert pos2_a != pos2_b, "Player 2 pylon position did not change after reset"
        
        dist_b = pos1_b.distance_to(pos2_b)
        assert dist_b >= flag_dist_min
        
    finally:
        env.close()

if __name__ == "__main__":
    test_ctf_pylon_counts_and_positions()

import pytest
import math
from src.utils.spawning_utils import SpawningUtils
from sc2.position import Point2

def test_calculate_safe_distance_scaling():
    # Large map
    dist = SpawningUtils.calculate_safe_distance(100, 100)
    # Diagonal is ~141.4. 40% is ~56.5.
    assert 56.0 < dist < 57.0
    
    # Small map where diagonal 40% might exceed bounds
    dist_small = SpawningUtils.calculate_safe_distance(20, 20)
    # diagonal 40% is ~11.3. max possible is (20-10)*0.8 = 8.0.
    assert dist_small == 8.0

def test_calculate_safe_distance_explicit():
    dist = SpawningUtils.calculate_safe_distance(100, 100, requested_dist=30.0)
    assert dist == 30.0

def test_get_random_pylon_positions_bounds():
    w, h = 100, 100
    margin = 5.0
    for _ in range(100):
        pos1, pos2 = SpawningUtils.get_random_pylon_positions(w, h, 20.0, margin=margin)
        
        for pos in [pos1, pos2]:
            assert margin <= pos.x <= w - margin
            assert margin <= pos.y <= h - margin

def test_get_random_pylon_positions_distance():
    w, h = 100, 100
    min_dist = 40.0
    for _ in range(100):
        pos1, pos2 = SpawningUtils.get_random_pylon_positions(w, h, min_dist)
        # Note: logic uses check_dist = min_dist - 1.0
        assert pos1.distance_to(pos2) >= min_dist - 1.0

def test_get_random_pylon_positions_fallback():
    # Impossible distance
    w, h = 20, 20
    pos1, pos2 = SpawningUtils.get_random_pylon_positions(w, h, 100.0, max_attempts=10)
    
    # Should be opposite corners (5,5) and (15,15) or (5,15) and (15,5)
    corners = { (5.0, 5.0), (15.0, 15.0), (5.0, 15.0), (15.0, 5.0) }
    assert (pos1.x, pos1.y) in corners
    assert (pos2.x, pos2.y) in corners
    assert pos1 != pos2

import pytest
import numpy as np
from src.utils.perspective_transformer import PerspectiveTransformer

def test_flip_observation():
    # 2 layers, 4x4
    obs = np.zeros((2, 4, 4), dtype=np.uint8)
    obs[0, 0, 0] = 1  # Top-Left
    obs[1, 3, 3] = 2  # Bottom-Right
    
    flipped = PerspectiveTransformer.flip_observation(obs)
    
    # After 180 deg rotation:
    # Top-Left (0,0) -> Bottom-Right (3,3)
    # Bottom-Right (3,3) -> Top-Left (0,0)
    assert flipped[0, 3, 3] == 1
    assert flipped[1, 0, 0] == 2
    assert flipped.shape == (2, 4, 4)

def test_flip_action_idx_single_head():
    grid_size = 4
    # (0,0) -> idx 0
    # (3,3) -> idx 15
    assert PerspectiveTransformer.flip_action_idx(0, grid_size) == 15
    assert PerspectiveTransformer.flip_action_idx(15, grid_size) == 0
    
    # (0,1) -> idx 1
    # (3,2) -> idx 14
    assert PerspectiveTransformer.flip_action_idx(1, grid_size) == 14

def test_flip_action_idx_dual_head():
    grid_size = 4
    grid_area = 16
    
    # Move Head
    assert PerspectiveTransformer.flip_action_idx(0, grid_size) == 15
    
    # Attack Head (indices 16-31)
    # Action 16 is Attack(0,0) -> should become Attack(3,3) which is 16 + 15 = 31
    assert PerspectiveTransformer.flip_action_idx(16, grid_size) == 31
    assert PerspectiveTransformer.flip_action_idx(31, grid_size) == 16

def test_flip_mask():
    grid_size = 2
    # [True, False, 
    #  False, False]
    mask = np.array([True, False, False, False])
    
    flipped = PerspectiveTransformer.flip_mask(mask, grid_size)
    
    # Should be [False, False,
    #            False, True]
    assert np.array_equal(flipped, np.array([False, False, False, True]))


def test_decode_world_spatial_idx_is_identity_for_all_owners():
    for owner in (1, 2):
        assert PerspectiveTransformer.decode_world_spatial_idx("move_global", 0, owner) == 0
        assert PerspectiveTransformer.decode_world_spatial_idx("attack_local", 512, owner) == 512
        assert PerspectiveTransformer.decode_world_spatial_idx("move_global", 1023, owner) == 1023

if __name__ == "__main__":
    # Manual run if needed
    test_flip_observation()
    test_flip_action_idx_single_head()
    test_flip_action_idx_dual_head()
    test_flip_mask()
    test_decode_world_spatial_idx_is_identity_for_all_owners()
    print("Tests passed!")

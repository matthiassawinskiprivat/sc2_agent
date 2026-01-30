import pytest
import os
import numpy as np
from imitation_learning.buffer import ImitationReplayBuffer

def test_imitation_buffer_loading(tmp_path):
    # Setup mock structure
    level_name = "test_level"
    trans_dir = tmp_path / "imitation_learning" / "replays_translated" / level_name
    trans_dir.mkdir(parents=True)
    
    # Create 2 mock datasets
    obs1 = np.ones((5, 10, 128, 128), dtype=np.uint8)
    act1 = np.arange(5, dtype=np.int32)
    np.savez(trans_dir / "data1.npz", observations=obs1, actions=act1)
    
    obs2 = np.zeros((3, 10, 128, 128), dtype=np.uint8)
    act2 = np.arange(5, 8, dtype=np.int32)
    np.savez(trans_dir / "data2.npz", observations=obs2, actions=act2)
    
    buffer = ImitationReplayBuffer(level_name, project_root=str(tmp_path))
    
    assert buffer.total_size == 8
    assert buffer.observations.shape == (8, 10, 128, 128)
    assert buffer.actions.shape == (8,)

def test_imitation_buffer_sampling(tmp_path):
    level_name = "test_level"
    trans_dir = tmp_path / "imitation_learning" / "replays_translated" / level_name
    trans_dir.mkdir(parents=True)
    
    obs = np.random.randint(0, 255, (100, 10, 128, 128), dtype=np.uint8)
    act = np.random.randint(0, 1000, (100,), dtype=np.int32)
    np.savez(trans_dir / "data.npz", observations=obs, actions=act)
    
    buffer = ImitationReplayBuffer(level_name, project_root=str(tmp_path))
    
    batch_obs, batch_act = buffer.sample_batch(32)
    assert batch_obs.shape == (32, 10, 128, 128)
    assert batch_act.shape == (32,)
    
    # Check if samples are from the original set
    for i in range(32):
        # Find match in original actions
        matches = np.where(act == batch_act[i])[0]
        assert len(matches) > 0

import pytest
import numpy as np
from imitation_learning.translator import ReplayTranslator

def test_process_raw_data():
    translator = ReplayTranslator()
    
    # Mock config
    config = {
        "env": {
            "action_heads": ["move", "attack"],
            "grid_size": 128
        }
    }
    
    # Mock raw recorded data (1 frame with 1 group action)
    mock_obs = np.zeros((10, 128, 128), dtype=np.uint8)
    recorded_data = [
        {
            "observation": mock_obs,
            "actions": [
                {
                    "ability_id": 16, # MOVE
                    "unit_tags": [1001, 1002, 1003], # 3 units selected
                    "target_world_pos": (50.0, 60.0),
                    "target_unit_tag": None
                }
            ]
        }
    ]
    
    pairs = translator.process_raw_data(recorded_data, config)
    
    # Should decompose into 3 examples
    assert len(pairs) == 3
    
    # Check action calculation
    # head_idx = 0 (move)
    # spatial_idx = 60 * 128 + 50 = 7680 + 50 = 7730
    # agent_action = 0 * 16384 + 7730 = 7730
    assert pairs[0]["action"] == 7730
    assert np.array_equal(pairs[0]["observation"], mock_obs)

def test_process_raw_data_attack():
    translator = ReplayTranslator()
    config = {
        "env": {
            "action_heads": ["move", "attack"],
            "grid_size": 128
        }
    }
    
    mock_obs = np.zeros((10, 128, 128), dtype=np.uint8)
    recorded_data = [
        {
            "observation": mock_obs,
            "actions": [
                {
                    "ability_id": 23, # ATTACK
                    "unit_tags": [1001],
                    "target_world_pos": (10.0, 10.0),
                    "target_unit_tag": None
                }
            ]
        }
    ]
    
    pairs = translator.process_raw_data(recorded_data, config)
    
    # head_idx = 1 (attack)
    # spatial_idx = 10 * 128 + 10 = 1290
    # agent_action = 1 * 16384 + 1290 = 17674
    assert pairs[0]["action"] == 17674

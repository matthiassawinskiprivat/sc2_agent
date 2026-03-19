
import numpy as np
import torch
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution

def test_distribution():
    action_dims = [2, 1024]
    dist = MaskableMultiCategoricalDistribution(action_dims)
    
    # Simulate flat mask from 4 envs:
    # 4 envs * 1026 = 4104 elements.
    mask = torch.ones((4, 1026), dtype=torch.bool)
    
    try:
        dist.apply_masking(mask)
        print("Success with shape:", mask.shape)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_distribution()

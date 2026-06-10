"""CUDA-safe universal policy sampling when logits or masks are degenerate."""
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import get_schedule_fn

from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from train.squad_batch_inference import sanitize_flat_action_mask


def _policy():
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(0, 1, (2, 32, 32), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
        }
    )
    act_space = spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32])
    policy = UniversalRecurrentPolicy(
        obs_space,
        act_space,
        get_schedule_fn(3e-4),
        rnn_type="gru",
        lstm_hidden_size=32,
        features_extractor_class=UnitControlNetwork,
        features_extractor_kwargs={
            "features_dim": 64,
            "thinking_steps": 1,
            "agent_architecture": "universal",
        },
    )
    policy.action_dist = MaskableMultiCategoricalDistribution(act_space.nvec)
    return policy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nan_logits_sampling_does_not_cuda_assert():
    policy = _policy().to("cuda")
    policy.set_training_mode(False)
    batch = 3
    obs = {
        "map": torch.rand(batch, 4, 128, 128, device="cuda"),
        "tactical": torch.rand(batch, 2, 32, 32, device="cuda"),
        "command": torch.rand(batch, 10, device="cuda"),
    }
    h = torch.zeros(1, batch, 32, device="cuda")
    lstm = RNNStates((h,), (h,))
    starts = torch.zeros(batch, device="cuda")
    masks = torch.ones(batch, FIXED_UNIVERSAL_COMMAND_DIM + 32 * 32, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        policy.command_head.weight.data.fill_(float("nan"))
        actions, _, logp, _ = policy.forward(obs, lstm, starts, action_masks=masks)
        torch.cuda.synchronize()
    assert actions.shape == (batch, 2)
    assert torch.isfinite(logp).all()


def test_sanitize_flat_action_mask_enables_head_zero():
    mask = np.zeros(FIXED_UNIVERSAL_COMMAND_DIM + 32 * 32, dtype=bool)
    fixed = sanitize_flat_action_mask(mask)
    assert fixed[0]
    assert not fixed[1:64].any()

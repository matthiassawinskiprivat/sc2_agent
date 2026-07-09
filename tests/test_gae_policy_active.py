"""GAE must skip buffer timesteps where policy_active is zero for that env column."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.agent.utils.disk_buffer import DiskDictRolloutBuffer


def _make_buffer(n_envs: int, buffer_size: int) -> DiskDictRolloutBuffer:
    obs_space = spaces.Dict(
        {
            "command": spaces.Box(0, 1, shape=(4,), dtype=np.float32),
        }
    )
    return DiskDictRolloutBuffer(
        buffer_size,
        obs_space,
        spaces.MultiDiscrete([2, 4]),
        device="cpu",
        n_envs=n_envs,
        temp_dir="C:/temp/sc2_gae_policy_active_test",
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=8,
    )


def test_gae_skips_inactive_env_slots_in_interleaved_rows():
    """
    num_envs=2 with one active env per row (legacy interleaved layout).

    Env0 real transitions sit at t=0,2; zeros at t=1 must not corrupt bootstrap.
    """
    buf = _make_buffer(n_envs=2, buffer_size=3)
    buf.pos = 3
    buf.full = True

    # t=0 env0 active, t=1 env1 active, t=2 env0 active
    buf.policy_active = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    buf.rewards = np.array(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32
    )
    buf.values = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32
    )
    buf.episode_starts = np.zeros((3, 2), dtype=np.float32)

    buf.compute_returns_and_advantage(torch.zeros((2, 1)), np.array([False, False]))

    # Env0: chain t=2 -> t=0 with gamma=0.99, lambda=1, values=0
    # t=2: gae = 1.0
    # t=0: gae = 1.0 + 0.99 * 1.0 = 1.99
    assert buf.advantages[2, 0] == pytest.approx(1.0)
    assert buf.advantages[0, 0] == pytest.approx(1.99)
    assert buf.advantages[1, 0] == 0.0

    # Env1: single active step at t=1
    assert buf.advantages[1, 1] == pytest.approx(2.0)
    assert buf.advantages[0, 1] == 0.0


def test_need_forward_writes_one_row_for_all_envs():
    """Batched vec-env forward must produce one buffer row, not one per env."""
    from unittest.mock import MagicMock

    from train.squad_batch_inference import SquadBatchRolloutHelper

    vec = MagicMock()
    vec.action_space = spaces.MultiDiscrete([4, 16])
    vec.env_method = MagicMock(
        side_effect=lambda name, *a, **kw: {
            "prepare_policy_batch": [None, None],
            "action_masks": [
                np.ones(20, dtype=bool),
                np.ones(20, dtype=bool),
            ],
            "get_obs_focus_unit_tag": [10, 20],
        }.get(name, [None])
    )

    policy = MagicMock()
    device = torch.device("cpu")
    act = torch.tensor([[0, 3], [1, 7]], dtype=torch.long)
    val = torch.tensor([[0.5], [0.6]])
    logp = torch.tensor([-0.2, -0.3])
    policy.obs_to_tensor = MagicMock(
        return_value=({"command": torch.zeros(2, 4)}, None)
    )
    policy.forward = MagicMock(return_value=(act, val, logp))

    model = MagicMock()
    model.device = device
    model.policy = policy

    helper = SquadBatchRolloutHelper(2)
    obs = {"command": np.zeros((2, 4), dtype=np.float32)}

    resolution = helper.resolve_rollout(
        vec_env=vec,
        num_envs=2,
        obs=obs,
        requires_policy=np.array([True, True]),
        model=model,
        is_recurrent=False,
        last_dones=np.array([False, False]),
        prev_focus_tags=np.array([10, 20]),
        lstm_states=None,
        device=device,
    )

    assert resolution.buffer_rows == []
    assert len(resolution.vec_rows) == 1
    assert resolution.vec_rows[0].env_indices == [0, 1]
    assert resolution.vec_rows[0].actions.shape == (2, 2)


def test_flat_buffer_get_respects_policy_active():
    buf = _make_buffer(n_envs=2, buffer_size=4)
    buf.enable_rnn_sampling = False
    buf.pos = 4
    buf.full = True
    buf.policy_active = np.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    buf.device = torch.device("cpu")
    batch = next(iter(buf.get(batch_size=8)))
    assert batch.observations["command"].shape[0] == 8

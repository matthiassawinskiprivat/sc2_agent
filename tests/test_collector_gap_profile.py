"""Collector-side gap timers (SC2_COLLECTOR_PROFILE)."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from train.squad_pointer_rollout import SquadPointerRolloutHelper


def test_pop_collector_side_profile_means():
    helper = SquadPointerRolloutHelper(1)
    helper._profile_flag = True
    helper.profile_add("policy_fwd", 0.01)
    helper.profile_add("policy_fwd", 0.03)
    means = helper.pop_collector_profile_means()
    assert means["collector/profile_policy_fwd_ms"] == 20.0
    assert helper.pop_collector_profile_means() == {}


def test_profile_add_is_off_without_flag():
    helper = SquadPointerRolloutHelper(1)
    helper._profile_flag = False
    helper.profile_add("step", 1.0)
    assert helper.pop_collector_profile_means() == {}


def test_resolve_rollout_records_policy_peek_and_opponent_forward():
    helper = SquadPointerRolloutHelper(1)
    helper._profile_flag = True

    vec = MagicMock()
    vec.action_space = spaces.MultiDiscrete([2, 2])
    vec.has_attr = MagicMock(return_value=True)
    peek_payload = {
        "obs": {"x": np.zeros((2,), dtype=np.float32)},
        "mask": np.ones(4, dtype=bool),
        "episode_start": False,
        "lstm": None,
        "pool_idx": 0,
    }
    vec.env_method = MagicMock(
        side_effect=lambda name, *args, **kwargs: {
            "peek_opponent_squad_pointer": [peek_payload],
            "set_pending_opponent_squad_actions": [None],
        }.get(name, [None])
    )

    model = MagicMock()
    model.device = torch.device("cpu")
    policy = MagicMock()
    model.policy = policy
    policy.obs_to_tensor = MagicMock(return_value=(torch.zeros(1, 3), True))
    policy.forward = MagicMock(
        return_value=(
            torch.zeros(1, 2),
            torch.zeros(1, 1),
            torch.zeros(1),
        )
    )
    policy.features_extractor = None

    with patch(
        "train.squad_pointer_rollout.opponent_pointer_vec_predict",
        return_value=(np.zeros((1, 4), dtype=np.int64), [None]),
    ) as opp_fwd:
        helper.resolve_rollout(
            vec_env=vec,
            num_envs=1,
            obs={"x": np.zeros((1, 3), dtype=np.float32)},
            requires_policy=np.array([True]),
            model=model,
            is_recurrent=False,
            last_dones=np.array([False]),
            prev_focus_tags=np.array([-1], dtype=np.int64),
            lstm_states=None,
            device=torch.device("cpu"),
            action_masks=np.ones((1, 4), dtype=bool),
        )
    opp_fwd.assert_called_once()
    means = helper.pop_collector_profile_means()
    assert "collector/profile_policy_fwd_ms" in means
    assert "collector/profile_peek_ms" in means
    assert "collector/profile_opp_fwd_ms" in means
    assert "collector/profile_peek_send_ms" in means

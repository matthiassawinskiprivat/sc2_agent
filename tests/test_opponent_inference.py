"""Tests for inference-only opponent loading and GPU cleanup."""
from unittest.mock import MagicMock, patch

import pytest
import torch
from gymnasium import spaces

from src.utils.cuda_memory import (
    clear_policy_optimizer,
    dispose_opponent_snapshot,
    dispose_ppo_model,
)
from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM
from src.utils.opponent_inference import (
    OpponentInferenceModel,
    build_opponent_policy,
    load_opponent_inference_checkpoint,
)


def _squad_obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(6, 128, 128), dtype=float),
            "command": spaces.Box(low=-1, high=1, shape=(10,), dtype=float),
            "anchor_tactical": spaces.Box(
                low=0, high=1, shape=(4, 32, 32), dtype=float
            ),
            "unit_scalars": spaces.Box(
                low=-1, high=1, shape=(24, SQUAD_UNIT_FEAT_DIM), dtype=float
            ),
            "unit_mask": spaces.Box(low=0, high=1, shape=(24,), dtype=float),
        }
    )


def test_build_opponent_policy_wires_maskable_multidiscrete_dist():
    from sb3_contrib.common.maskable.distributions import (
        MaskableMultiCategoricalDistribution,
    )

    from src.agent.models.squad_policy import SquadRecurrentPolicy
    from src.agent.squad_constants import squad_action_space_nvec

    act = spaces.MultiDiscrete(squad_action_space_nvec())
    policy = build_opponent_policy(
        SquadRecurrentPolicy,
        _squad_obs_space(),
        act,
        {
            "features_extractor_kwargs": {"features_dim": 64, "thinking_steps": 1},
            "lstm_hidden_size": 64,
        },
        torch.device("cpu"),
    )
    assert isinstance(policy.action_dist, MaskableMultiCategoricalDistribution)


def test_build_opponent_policy_has_no_optimizer():
    from src.agent.models.squad_network import SquadNetwork
    from src.agent.models.squad_policy import SquadRecurrentPolicy
    from src.agent.squad_constants import squad_action_space_nvec

    policy = build_opponent_policy(
        SquadRecurrentPolicy,
        _squad_obs_space(),
        spaces.MultiDiscrete(squad_action_space_nvec()),
        {
            "features_extractor_class": SquadNetwork,
            "features_extractor_kwargs": {"features_dim": 64, "thinking_steps": 1},
            "lstm_hidden_size": 64,
        },
        torch.device("cpu"),
    )
    assert getattr(policy, "optimizer", None) is None


def test_opponent_inference_model_predict_delegates():
    policy = MagicMock()
    policy.parameters = MagicMock(
        return_value=iter([torch.zeros(1, requires_grad=False)])
    )
    policy.predict.return_value = (torch.zeros(2), None)
    model = OpponentInferenceModel(policy=policy)
    assert model._inference_only is True
    action, state = model.predict({"map": 1}, deterministic=True)
    policy.predict.assert_called_once()
    assert state is None


def test_dispose_opponent_snapshot_clears_policy_ref():
    policy = MagicMock()
    policy.parameters = MagicMock(return_value=iter([]))
    model = OpponentInferenceModel(policy=policy)
    dispose_opponent_snapshot(model, empty_cache=False)
    assert model.policy is None
    policy.set_training_mode.assert_called_once_with(False)
    policy.to.assert_called_once_with("cpu")


def test_dispose_ppo_model_clears_optimizer_before_delete():
    policy = MagicMock()
    opt = MagicMock()
    policy.optimizer = opt
    model = MagicMock()
    model._inference_only = False
    model.policy = policy
    model.rollout_buffer = None
    model._last_lstm_states = MagicMock()

    dispose_ppo_model(model, empty_cache=False)

    assert policy.optimizer is None
    opt.state.clear.assert_called_once()
    policy.to.assert_called_once_with("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_clear_policy_optimizer_frees_gpu_state():
    class _Policy:
        pass

    policy = _Policy()
    linear = torch.nn.Linear(4, 2, device="cuda:0")
    policy.optimizer = torch.optim.Adam(linear.parameters())
    loss = linear(torch.randn(2, 4, device="cuda:0")).sum()
    loss.backward()
    policy.optimizer.step()
    assert len(policy.optimizer.state) > 0
    clear_policy_optimizer(policy)
    assert getattr(policy, "optimizer", None) is None


def test_load_opponent_inference_checkpoint_strict_path(tmp_path):
    from src.agent.models.squad_policy import SquadRecurrentPolicy

    obs = _squad_obs_space()
    act = spaces.MultiDiscrete([2] * 24 + [64, 1024])
    ckpt = tmp_path / "opp.zip"
    with patch(
        "src.utils.model_utils.load_policy_weights_from_zip",
        return_value=(True, [], []),
    ) as mock_load:
        snap = load_opponent_inference_checkpoint(
            str(ckpt),
            agent_architecture="squad",
            observation_space=obs,
            action_space=act,
            policy_class=SquadRecurrentPolicy,
            policy_kwargs={
                "features_extractor_kwargs": {"features_dim": 64, "thinking_steps": 1},
                "lstm_hidden_size": 64,
            },
            rnn_type="gru",
            device=torch.device("cpu"),
        )
    mock_load.assert_called_once()
    assert isinstance(snap, OpponentInferenceModel)
    assert getattr(snap.policy, "optimizer", None) is None

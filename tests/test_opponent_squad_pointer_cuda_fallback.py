"""Opponent squad-pointer predict CUDA fault handling.

CUDA poison errors ("illegal memory access", "unspecified launch failure", …) are
re-raised so the SubprocVecEnv worker handler calls os._exit(1) and the collector
rebuilds with a fresh GPU context.  Silently swallowing them caused silent training
degradation: the opponent used a random fallback on every step for the rest of the
run while the log only showed one line.

Non-CUDA runtime errors continue to use the safe fallback so an isolated forward
failure does not kill the worker.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_env(unit_mask):
    """Return a minimal CaptureTheFlagSingleEnv stub for opponent-predict tests."""
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
    from src.agent.squad_constants import squad_action_space_nvec

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.bot = MagicMock()
    env.bot.squad_action_masks = MagicMock(
        return_value=np.ones(int(sum(squad_action_space_nvec())), dtype=bool)
    )
    env._get_squad_observation_for_pool = MagicMock(
        return_value={"unit_mask": unit_mask}
    )
    env._ensure_opponent_model_loaded = MagicMock()
    env._log = MagicMock()
    env._opponent_lstm_state = ("stale",)
    env._opponent_infer_source = "frozen"
    env.opponent_model = MagicMock()
    return env


def _make_opp(side_effect):
    opp = MagicMock()
    opp.policy.set_training_mode = MagicMock()
    opp.policy.features_extractor = MagicMock()
    opp.policy.features_extractor.clear_forward_cache = MagicMock()
    opp.predict = MagicMock(side_effect=side_effect)
    return opp


@pytest.mark.parametrize("error_msg", [
    "CUDA error: unspecified launch failure",
    "CUDA error: an illegal memory access was encountered",
    "CUDA out of memory",
])
def test_opponent_predict_cuda_poison_propagates(error_msg):
    """CUDA poison errors must re-raise so the worker exits and rebuilds."""
    import asyncio

    unit_mask = np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32)
    env = _make_env(unit_mask)
    env._resolve_opponent_model = MagicMock(
        return_value=(_make_opp(RuntimeError(error_msg)), "frozen")
    )

    with pytest.raises(RuntimeError, match="CUDA error|out of memory"):
        asyncio.run(env._predict_opponent_squad_pointer(0))


def test_opponent_predict_non_cuda_error_uses_fallback():
    """Non-CUDA runtime errors must NOT kill the worker — return a safe fallback."""
    import asyncio
    from src.agent.squad_constants import squad_fallback_action

    unit_mask = np.array([1.0, 1.0] + [0.0] * 22, dtype=np.float32)
    env = _make_env(unit_mask)
    env._resolve_opponent_model = MagicMock(
        return_value=(_make_opp(RuntimeError("model output shape mismatch")), "frozen")
    )

    out = asyncio.run(env._predict_opponent_squad_pointer(0))
    expected = squad_fallback_action(unit_mask)
    assert np.array_equal(out, expected)
    assert env._opponent_lstm_state is None
    env._log.assert_called()


def test_opponent_squad_pointer_sets_episode_start_when_state_cleared():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
    from src.agent.squad_constants import squad_action_space_nvec

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.bot = MagicMock()
    unit_mask = np.ones(24, dtype=np.float32)
    env.bot.squad_action_masks = MagicMock(
        return_value=np.ones(int(sum(squad_action_space_nvec())), dtype=bool)
    )
    env._get_squad_observation_for_pool = MagicMock(
        return_value={"unit_mask": unit_mask}
    )
    env._ensure_opponent_model_loaded = MagicMock()
    env._log = MagicMock()
    env._opponent_lstm_state = None
    env._opponent_infer_source = "frozen"
    env.opponent_model = MagicMock()

    captured = {}

    def _predict(obs, action_masks=None, **kwargs):
        captured.update(kwargs)
        n = len(squad_action_space_nvec())
        return np.zeros(n, dtype=np.int64), ("new_state",)

    opp = MagicMock()
    opp.policy.set_training_mode = MagicMock()
    opp.policy.features_extractor = MagicMock()
    opp.policy.features_extractor.clear_forward_cache = MagicMock()
    opp.predict = _predict
    env._resolve_opponent_model = MagicMock(return_value=(opp, "frozen"))

    import asyncio

    asyncio.run(env._predict_opponent_squad_pointer(1))
    assert captured.get("episode_start") is not None
    assert bool(np.asarray(captured["episode_start"]).reshape(-1)[0]) is True
    assert env._opponent_lstm_state == ("new_state",)

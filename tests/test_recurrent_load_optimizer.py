"""MaskableRecurrentPPO.load must honor restore_optimizer for same-level resume."""
from unittest.mock import MagicMock, patch

from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO


def test_recurrent_load_passes_optimizer_when_restore_optimizer_true():
    captured = {}

    def fake_super_load(cls, path, env=None, device="auto", custom_objects=None, **kwargs):
        captured["custom_objects"] = dict(custom_objects or {})
        return MagicMock()

    with patch.object(
        MaskableRecurrentPPO.__mro__[1],
        "load",
        new=fake_super_load,
    ), patch(
        "src.agent.maskable_recurrent_ppo.checkpoint_obs_action_mismatch_env",
        return_value=False,
    ):
        MaskableRecurrentPPO.load(
            "dummy.zip",
            env=MagicMock(),
            restore_optimizer=True,
        )

    assert "optimizer" not in captured["custom_objects"]


def test_recurrent_load_strips_optimizer_by_default():
    captured = {}

    def fake_super_load(cls, path, env=None, device="auto", custom_objects=None, **kwargs):
        captured["custom_objects"] = dict(custom_objects or {})
        return MagicMock()

    with patch.object(
        MaskableRecurrentPPO.__mro__[1],
        "load",
        new=fake_super_load,
    ), patch(
        "src.agent.maskable_recurrent_ppo.checkpoint_obs_action_mismatch_env",
        return_value=False,
    ):
        MaskableRecurrentPPO.load("dummy.zip", env=MagicMock())

    assert captured["custom_objects"].get("optimizer") is None

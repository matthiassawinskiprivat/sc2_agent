"""MaskableRecurrentPPO.load must honor restore_optimizer for same-level resume."""
from unittest.mock import MagicMock, patch

from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.models.squad_network import SquadNetwork
from src.agent.models.squad_policy import SquadRecurrentPolicy
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy


def _patch_super_load(captured):
    """Replace parent ``load`` with a classmethod that records ``custom_objects``."""

    @classmethod
    def fake_super_load(cls, path, env=None, device="auto", custom_objects=None, **kwargs):
        captured["custom_objects"] = dict(custom_objects or {})
        return MagicMock()

    return patch.object(MaskableRecurrentPPO.__mro__[1], "load", new=fake_super_load)


def test_recurrent_load_passes_optimizer_when_restore_optimizer_true():
    captured = {}

    with _patch_super_load(captured), patch(
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

    with _patch_super_load(captured), patch(
        "src.agent.maskable_recurrent_ppo.checkpoint_obs_action_mismatch_env",
        return_value=False,
    ):
        MaskableRecurrentPPO.load("dummy.zip", env=MagicMock())

    assert captured["custom_objects"].get("optimizer") is None


def test_recurrent_load_defaults_universal_policy_without_squad_hints():
    captured = {}

    with _patch_super_load(captured), patch(
        "src.agent.maskable_recurrent_ppo.checkpoint_obs_action_mismatch_env",
        return_value=False,
    ), patch.object(
        MaskableRecurrentPPO,
        "_default_policy_class_for_load",
        return_value=UniversalRecurrentPolicy,
    ):
        MaskableRecurrentPPO.load("dummy.zip", env=MagicMock())

    assert captured["custom_objects"]["policy_class"] is UniversalRecurrentPolicy


def test_recurrent_load_uses_squad_policy_when_squad_fe_in_kwargs():
    captured = {}

    with _patch_super_load(captured), patch(
        "src.agent.maskable_recurrent_ppo.checkpoint_obs_action_mismatch_env",
        return_value=False,
    ):
        MaskableRecurrentPPO.load(
            "dummy.zip",
            env=MagicMock(),
            policy_kwargs={
                "features_extractor_class": SquadNetwork,
                "features_extractor_kwargs": {"use_self_attention": True},
            },
        )

    assert captured["custom_objects"]["policy_class"] is SquadRecurrentPolicy
    fe_kw = captured["custom_objects"]["policy_kwargs"]["features_extractor_kwargs"]
    assert fe_kw["use_self_attention"] is True


def test_recurrent_load_honors_explicit_policy_override():
    captured = {}

    with _patch_super_load(captured), patch(
        "src.agent.maskable_recurrent_ppo.checkpoint_obs_action_mismatch_env",
        return_value=False,
    ):
        MaskableRecurrentPPO.load(
            "dummy.zip",
            env=MagicMock(),
            policy=SquadRecurrentPolicy,
        )

    assert captured["custom_objects"]["policy_class"] is SquadRecurrentPolicy

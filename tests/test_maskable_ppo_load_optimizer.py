"""MaskablePPO.load must honor restore_optimizer on recovery path."""
from unittest.mock import MagicMock, patch

from src.agent.maskable_ppo import MaskablePPO


def test_maskable_ppo_load_restores_optimizer_on_recovery_path():
    captured = {"calls": 0}

    def fake_super_load(*args, **kwargs):
        captured["custom_objects"] = dict(kwargs.get("custom_objects") or {})
        captured["calls"] += 1
        if captured["calls"] == 1:
            raise RuntimeError("shape mismatch")
        return MagicMock(policy=MagicMock(optimizer=MagicMock()))

    with patch.object(
        MaskablePPO.__mro__[1],
        "load",
        new=fake_super_load,
    ), patch(
        "src.utils.model_utils.load_optimizer_state_from_zip",
        return_value=(True, "ok"),
    ) as mock_restore:
        MaskablePPO.load("dummy.zip", env=MagicMock(), restore_optimizer=True)

    assert captured["custom_objects"].get("optimizer") is None
    mock_restore.assert_called_once()

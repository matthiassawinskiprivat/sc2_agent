import numpy as np

from src.agent.universal_unit_control_agent import UniversalUnitControlAgent


def test_universal_agent_returns_all_true_spatial_mask():
    agent = UniversalUnitControlAgent(grid_size=32)
    out = agent.get_action_mask(None)

    assert out.shape == (32 * 32,)
    assert np.array_equal(out, np.ones(32 * 32, dtype=bool))


def test_universal_agent_concat_mask_uses_all_true_spatial_part():
    agent = UniversalUnitControlAgent(grid_size=32)
    out = agent.get_action_mask(["move_local", "attack_local"])

    assert out.shape == (2 + 32 * 32,)
    assert np.array_equal(out[:2], np.ones(2, dtype=bool))
    assert np.array_equal(out[2:], np.ones(32 * 32, dtype=bool))

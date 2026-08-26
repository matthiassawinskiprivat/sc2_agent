import numpy as np
import pytest

from src.transformers.ui_transformer import RealTimeUITransformer


def test_command_assignment_visible_over_unit_density():
    """Protocol layers must overdraw density, not lose to np.maximum with white."""
    obs = np.zeros((2, 32, 32), dtype=np.float32)
    obs[0, 14:18, 14:18] = 1.0  # allied_unit_density
    obs[1, 14:18, 14:18] = 1.0  # command_assignment
    active = ["allied_unit_density", "command_assignment"]

    frame = RealTimeUITransformer(display_size=(512, 512)).update_display(
        obs, active, active_layers=active
    )

    center = frame[14 * 16 : 18 * 16, 14 * 16 : 18 * 16]
    unique = np.unique(center.reshape(-1, 3), axis=0)
    assert [255, 0, 255] in unique.tolist(), (
        "command_assignment should render magenta over white density"
    )


def test_ui_typed_a_dps_massive_displays_base_plus_bonus():
    """Debug UI must show total vs-massive threat, not washed-out bonus-only."""
    # Channels: a_dps_base (bonus-free), a_dps_massive (bonus-only tensor values).
    obs = np.zeros((2, 32, 32), dtype=np.float32)
    obs[0, 10:22, 10:22] = 0.15  # allied_a_dps_base
    obs[1, 10:22, 10:22] = 0.06  # allied_a_dps_massive (bonus-only)
    active = ["allied_a_dps_base", "allied_a_dps_massive"]

    ui = RealTimeUITransformer(display_size=(32, 32))
    # Isolate massive layer — display should use base+bonus intensity.
    frame = ui.update_display(obs, ["allied_a_dps_massive"], active_layers=active)
    color = np.array(ui.layer_colors["allied_a_dps_massive"], dtype=np.float32)
    expected = np.clip(0.15 + 0.06, 0.0, 1.0) * color * 255.0
    pixel = frame[16, 16].astype(np.float32)
    assert pixel == pytest.approx(expected, abs=1.0)

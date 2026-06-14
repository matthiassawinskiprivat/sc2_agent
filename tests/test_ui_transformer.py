import numpy as np

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

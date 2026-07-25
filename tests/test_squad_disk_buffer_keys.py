"""Squad pointer obs keys must use disk memmap IPC, not learner RAM allocation."""
import numpy as np
from gymnasium import spaces

from src.agent.squad_constants import MAX_SQUAD_UNITS, SQUAD_UNIT_FEAT_DIM
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer
from src.agent.utils.disk_buffer import is_disk_offloaded_obs_key
from src.utils.async_buffer_setup import configure_collector_static_buffer


def test_anchor_tactical_is_disk_offloaded_key():
    assert is_disk_offloaded_obs_key("anchor_tactical")
    assert is_disk_offloaded_obs_key("unit_tactical")  # legacy name still matches substring
    assert not is_disk_offloaded_obs_key("unit_scalars")
    assert not is_disk_offloaded_obs_key("unit_mask")
    assert not is_disk_offloaded_obs_key("command")


def test_configure_static_buffer_memmaps_anchor_tactical(tmp_path):
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (15, 128, 128), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
            "anchor_tactical": spaces.Box(0, 1, (15, 32, 32), dtype=np.float32),
            "unit_scalars": spaces.Box(
                0, 1, (MAX_SQUAD_UNITS, SQUAD_UNIT_FEAT_DIM), dtype=np.float32
            ),
            "unit_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
        }
    )
    buf = AsyncDiskBuffer(
        4,
        obs_space,
        spaces.MultiDiscrete([2] * MAX_SQUAD_UNITS + [FIXED_UNIVERSAL_COMMAND_DIM, 1024]),
        n_envs=1,
        temp_dir=str(tmp_path),
    )
    static = tmp_path / "buffer_0"
    configure_collector_static_buffer(buf, obs_space, str(static))

    assert isinstance(buf.observations["map"], np.memmap)
    assert isinstance(buf.observations["anchor_tactical"], np.memmap)
    assert not isinstance(buf.observations["command"], np.memmap)

    meta = {"buffer_size": 4, "n_envs": 1}
    buf.observations["anchor_tactical"][0, 0, 0, 0, 0] = 7.0
    buf.observations["anchor_tactical"].flush()

    learner = AsyncDiskBuffer(
        4,
        obs_space,
        spaces.MultiDiscrete([2] * MAX_SQUAD_UNITS + [FIXED_UNIVERSAL_COMMAND_DIM, 1024]),
        n_envs=1,
        temp_dir=str(tmp_path / "learner"),
    )
    AsyncDiskBuffer.attach_disk_observations(learner, str(static), meta)
    assert learner.observations["anchor_tactical"][0, 0, 0, 0, 0] == 7.0

"""AsyncSharedRamBuffer.reset must not detach map/tactical from shared memory."""
import numpy as np
import pytest
from gymnasium import spaces

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.utils.async_shared_ram_buffer import AsyncSharedRamBuffer


def _obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(0, 1, (4, 128, 128), dtype=np.float32),
            "tactical": spaces.Box(0, 1, (2, 32, 32), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
        }
    )


def test_reset_preserves_shared_memory_observation_views(tmp_path):
    buf = AsyncSharedRamBuffer(
        8,
        _obs_space(),
        spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32]),
        n_envs=2,
        temp_dir=str(tmp_path),
        rnn_type="gru",
        hidden_size=16,
    )
    buf.instance_id = "buffer_test"
    buf.instance_dir = str(tmp_path)
    buf.bind_shared_observation_buffers()

    map_before = buf.observations["map"]
    map_before[0, 0, 0, 0, 0] = 42.0
    base_before = map_before.base

    buf.reset()

    map_after = buf.observations["map"]
    assert map_after.base is base_before, "reset() must not replace shm-backed map array"
    map_after[1, 1, 1, 0, 0] = 99.0

    meta = {"shared_obs": {}}
    for key in buf.shared_observation_keys():
        meta["shared_obs"][key] = {
            "name": buf.shm_segment_name(key),
            "shape": list(buf.observations[key].shape),
        }

    learner = AsyncSharedRamBuffer(
        8,
        _obs_space(),
        spaces.MultiDiscrete([FIXED_UNIVERSAL_COMMAND_DIM, 32 * 32]),
        n_envs=2,
        temp_dir=str(tmp_path / "learner"),
        rnn_type="gru",
        hidden_size=16,
    )
    learner.observations = {}
    AsyncSharedRamBuffer.attach_shared_observations(learner, meta)

    assert learner.observations["map"][0, 0, 0, 0, 0] == pytest.approx(42.0)
    assert learner.observations["map"][1, 1, 1, 0, 0] == pytest.approx(99.0)

    buf.close_shared_memory(unlink=True)
    learner.close_shared_memory(unlink=False)

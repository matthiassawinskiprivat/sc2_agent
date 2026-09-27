"""Shared-memory opponent peek: compact payload + reconstruct."""
from gymnasium import spaces
import numpy as np

from train.opponent_peek_ipc import (
    OpponentPeekSharedMemory,
    attach_peek_views,
    compact_opponent_peek_obs,
    reconstruct_opponent_peek_obs,
    write_peek_spatial,
)


def _tiny_space():
    return spaces.Dict(
        {
            "map": spaces.Box(0, 1, shape=(2, 4, 4), dtype=np.float32),
            "anchor_tactical": spaces.Box(0, 1, shape=(2, 4, 4), dtype=np.float32),
            "command": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        }
    )


def _full_obs(tag: float) -> dict:
    return {
        "map": np.full((2, 4, 4), tag, dtype=np.float32),
        "anchor_tactical": np.full((2, 4, 4), tag + 0.25, dtype=np.float32),
        "command": np.array([tag, 0.0, 1.0], dtype=np.float32),
        "unit_mask": np.ones(2, dtype=np.float32),
    }


def test_write_and_reconstruct_roundtrip():
    shm = OpponentPeekSharedMemory(2, _tiny_space(), instance_id="peektest_rt")
    try:
        views = attach_peek_views(shm.worker_meta())
        obs0 = _full_obs(1.0)
        obs1 = _full_obs(3.0)
        means0 = write_peek_spatial(views, 0, obs0)
        means1 = write_peek_spatial(views, 1, obs1)
        assert means0 is not None and means1 is not None
        compact0 = compact_opponent_peek_obs(obs0, slot=0, sample_means=means0)
        compact1 = compact_opponent_peek_obs(obs1, slot=1, sample_means=means1)
        assert "map" not in compact0
        assert compact0["command"][0] == 1.0
        # Cached pool obs must keep its spatial tensors.
        assert "map" in obs0
        got0 = reconstruct_opponent_peek_obs({"obs": compact0}, shm, 0)
        got1 = reconstruct_opponent_peek_obs({"obs": compact1}, shm, 1)
        assert np.allclose(got0["map"], obs0["map"])
        assert np.allclose(got1["anchor_tactical"], obs1["anchor_tactical"])
        assert got0["command"][0] == 1.0
    finally:
        shm.close(unlink=True)


def test_reconstruct_full_pickle_fallback_without_shm():
    obs = _full_obs(2.0)
    got = reconstruct_opponent_peek_obs({"obs": obs}, None, 0)
    assert np.allclose(got["map"], obs["map"])


def test_fingerprint_mismatch_skips_env():
    shm = OpponentPeekSharedMemory(1, _tiny_space(), instance_id="peektest_fp")
    try:
        views = attach_peek_views(shm.worker_meta())
        obs = _full_obs(4.0)
        means = write_peek_spatial(views, 0, obs)
        compact = compact_opponent_peek_obs(obs, slot=0, sample_means=means)
        compact["_peek_means"] = {"map": 99.0, "anchor_tactical": 99.0}
        assert reconstruct_opponent_peek_obs({"obs": compact}, shm, 0) is None
    finally:
        shm.close(unlink=True)


def test_env_peek_compacts_when_shm_attached():
    from unittest.mock import MagicMock
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    shm = OpponentPeekSharedMemory(1, _tiny_space(), instance_id="peektest_env")
    try:
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
        env.worker_index = 0
        env._log = MagicMock()
        env._uses_squad_pointer = MagicMock(return_value=True)
        env._preview_computer_opponent = MagicMock(return_value=False)
        env.bot = MagicMock()
        env.bot.squad_action_masks = MagicMock(return_value=np.ones(4, dtype=bool))
        env.current_trainee_pool_index = 0
        env._paired_opponent_pool_index = MagicMock(return_value=1)
        env._pool_alive_tags = MagicMock(return_value=[7])
        env._squad_obs_step_cache = {}
        env._opponent_lstm_states = {}
        full = _full_obs(5.0)
        env._get_squad_observation_for_pool = MagicMock(return_value=full)
        assert env.attach_opponent_peek_shm(shm.worker_meta()) is True
        payload = env.peek_opponent_squad_pointer()
        assert payload["obs"].get("_peek_shm") is True
        assert "map" not in payload["obs"]
        assert "map" in full
        rebuilt = reconstruct_opponent_peek_obs(payload, shm, 0)
        assert np.allclose(rebuilt["map"], full["map"])
    finally:
        shm.close(unlink=True)


def test_set_pending_actions_reads_worker_slot():
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv

    env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
    env.worker_index = 1
    env._pending_opponent_squad_action = None
    env._opponent_lstm_states = {}
    packed = [
        None,
        (np.array([1, 2, 3], dtype=np.int64), "lstm-b", 2),
    ]
    env.set_pending_opponent_squad_actions(packed)
    assert np.array_equal(env._pending_opponent_squad_action, [1, 2, 3])
    assert env._opponent_lstm_states[2] == "lstm-b"


def test_hydrate_vec_step_obs_from_placeholders():
    from train.opponent_peek_ipc import compact_step_obs, hydrate_vec_spatial_obs, write_peek_spatial

    shm = OpponentPeekSharedMemory(2, _tiny_space(), instance_id="steptest_hy")
    try:
        views = attach_peek_views(shm.worker_meta())
        obs0 = _full_obs(2.0)
        obs1 = _full_obs(8.0)
        means0 = write_peek_spatial(views, 0, obs0)
        means1 = write_peek_spatial(views, 1, obs1)
        compact0 = compact_step_obs(obs0)
        compact1 = compact_step_obs(obs1)
        assert compact0["map"].shape == (1,)
        assert "command" in compact0
        stacked = {
            "map": np.stack([compact0["map"], compact1["map"]]),
            "command": np.stack([compact0["command"], compact1["command"]]),
        }
        infos = [
            {"spatial_shm": True, "spatial_shm_slot": 0, "spatial_shm_means": means0},
            {"spatial_shm": True, "spatial_shm_slot": 1, "spatial_shm_means": means1},
        ]
        got = hydrate_vec_spatial_obs(stacked, infos, shm, 2)
        assert got["map"].shape == (2, 2, 4, 4)
        assert np.allclose(got["map"][0], obs0["map"])
        assert np.allclose(got["map"][1], obs1["map"])
        assert np.allclose(got["command"][1], obs1["command"])
    finally:
        shm.close(unlink=True)


def test_hydrate_uses_reset_infos_when_done():
    from train.opponent_peek_ipc import compact_step_obs, hydrate_vec_spatial_obs, write_peek_spatial

    shm = OpponentPeekSharedMemory(1, _tiny_space(), instance_id="steptest_done")
    try:
        views = attach_peek_views(shm.worker_meta())
        terminal = _full_obs(1.0)
        reset_obs = _full_obs(9.0)
        write_peek_spatial(views, 0, terminal)
        means_reset = write_peek_spatial(views, 0, reset_obs)
        compact = compact_step_obs(reset_obs)
        stacked = {"map": compact["map"][None, ...], "command": compact["command"][None, ...]}
        stale = {"spatial_shm": True, "spatial_shm_slot": 0, "spatial_shm_means": {"map": 0.0}}
        reset_info = {
            "spatial_shm": True,
            "spatial_shm_slot": 0,
            "spatial_shm_means": means_reset,
        }
        got = hydrate_vec_spatial_obs(
            stacked,
            [stale],
            shm,
            1,
            reset_infos=[reset_info],
            dones=[True],
        )
        assert np.allclose(got["map"][0], reset_obs["map"])
    finally:
        shm.close(unlink=True)


def test_export_squad_obs_compacts_only_for_subproc():
    from unittest.mock import MagicMock
    from train.envs.capture_the_flag_single_env import CaptureTheFlagSingleEnv
    from train.opponent_peek_ipc import hydrate_vec_spatial_obs

    shm = OpponentPeekSharedMemory(1, _tiny_space(), instance_id="steptest_exp")
    try:
        env = CaptureTheFlagSingleEnv.__new__(CaptureTheFlagSingleEnv)
        env.worker_index = 0
        env.num_workers = 2
        env._log = MagicMock()
        env._uses_squad_pointer = MagicMock(return_value=True)
        env.requires_policy_action = MagicMock(return_value=True)
        env._cached_squad_action_masks = MagicMock(return_value=None)
        env.attach_opponent_peek_shm(shm.worker_meta())
        full = _full_obs(6.0)
        pipe_obs, info = env._export_squad_obs(full, {})
        assert pipe_obs["map"].shape == (1,)
        assert "map" in full and full["map"].shape == (2, 4, 4)
        assert info.get("spatial_shm") is True
        stacked = {"map": pipe_obs["map"][None, ...], "command": pipe_obs["command"][None, ...]}
        got = hydrate_vec_spatial_obs(stacked, [info], shm, 1)
        assert np.allclose(got["map"][0], full["map"])

        env.num_workers = 1
        kept, info2 = env._export_squad_obs(full, {})
        assert kept["map"].shape == (2, 4, 4)
        assert not info2.get("spatial_shm")
    finally:
        shm.close(unlink=True)

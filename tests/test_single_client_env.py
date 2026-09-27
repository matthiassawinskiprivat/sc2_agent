
import pytest
import asyncio
import numpy as np
from train.envs.single_client_env import SingleClientEnv
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.fixture
def mock_sc2_process():
    process = MagicMock()
    process.__aenter__ = AsyncMock(return_value=MagicMock())
    process.__aexit__ = AsyncMock()
    process.ws = MagicMock()
    return process

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.join_game = AsyncMock()
    client.step = AsyncMock()
    return client

def test_single_client_env_init():
    # This should fail because SingleClientEnv doesn't exist yet
    env = SingleClientEnv()
    assert env.worker_index == 0

def test_single_client_env_reset(mock_sc2_process, mock_client):
    # Mocking dependencies
    env = SingleClientEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda **kwargs: mock_client
    )
    
    # We need to mock _setup_game_session and _get_joint_observation 
    # as they are usually implemented by subclasses
    env._setup_game_session = AsyncMock()
    env._get_joint_observation = AsyncMock(return_value={"obs": "data"})
    
    # Mock launch_process on manager
    env._manager.launch_process = AsyncMock()
    
    # We need to mock run_until_complete on the loop to just return the coro result
    # since we can't easily run it while pytest-asyncio is active.
    # Alternatively, just don't use pytest.mark.asyncio for this test.
    
    obs, info = env.reset()
    
    assert obs == {"obs": "data"}
    env._setup_game_session.assert_called_once()
    env._manager.launch_process.assert_called_once()


def test_single_client_env_reset_retries_after_startup_failure(mock_sc2_process, mock_client):
    env = SingleClientEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda **kwargs: mock_client,
        training_conf={"env": {"startup_max_retries": 3}},
    )
    env._get_joint_observation = AsyncMock(return_value={"obs": "data"})

    call_count = {"n": 0}

    async def flaky_setup():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("create_game failed")

    env._setup_game_session = flaky_setup
    env._manager.launch_process = AsyncMock()
    env.close = MagicMock()

    with patch("train.envs.single_client_env.time.sleep"):
        obs, info = env.reset()

    assert obs == {"obs": "data"}
    assert call_count["n"] == 2
    assert env._manager.launch_process.call_count == 2
    assert env.close.call_count == 1


def test_single_client_env_startup_timeout_scales_with_worker_index():
    env0 = SingleClientEnv(worker_index=0, num_workers=5, training_conf={"env": {}})
    env4 = SingleClientEnv(worker_index=4, num_workers=5, training_conf={"env": {}})
    assert env4._startup_timeout_sec() > env0._startup_timeout_sec()
    # Default stagger 2.5s + 8s join-slack per later worker.
    assert env4._startup_timeout_sec() - env0._startup_timeout_sec() == pytest.approx(42.0)
    # Base + in_game poll sit in the budget even for worker 0.
    assert env0._startup_timeout_sec() == pytest.approx(105.0)


def test_single_client_env_accepts_faster_startup_knobs():
    env = SingleClientEnv(
        worker_index=2,
        num_workers=6,
        training_conf={
            "env": {
                "sc2_startup_stagger_sec": 0.5,
                "sc2_post_create_delay_sec": 0.25,
                "sc2_close_settle_sec": 0.1,
                "sc2_process_tree_wait_sec": 0.25,
            }
        },
    )
    assert env._manager._startup_stagger_sec() == pytest.approx(0.5)
    assert env._manager.post_create_delay_sec == pytest.approx(0.25)
    assert env._manager.close_settle_sec == pytest.approx(0.1)
    assert env._manager.process_tree_wait_sec == pytest.approx(0.25)
    # worker 2: 2×0.5s stagger + 2×8s join-slack; 90s base + 15s in_game.
    assert env._startup_timeout_sec() == pytest.approx(90.0 + 15.0 + 1.0 + 16.0)


def test_single_client_env_reads_sc2_timing_from_pipeline_yaml(tmp_path):
    """pipeline.yaml ``env`` is the shared home for SC2 launch / hard-reset knobs."""
    (tmp_path / "pipeline.yaml").write_text(
        "\n".join(
            [
                "levels: []",
                "env:",
                "  sc2_startup_stagger_sec: 1.25",
                "  sc2_post_create_delay_sec: 0.4",
                "  sc2_close_settle_sec: 0.2",
                "  sc2_process_tree_wait_sec: 0.3",
                "  sc2_retry_backoff_sec: 1.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = SingleClientEnv(
        worker_index=3,
        num_workers=4,
        training_conf={"config_dir": str(tmp_path)},
    )
    assert env._manager._startup_stagger_sec() == pytest.approx(1.25)
    assert env._manager.post_create_delay_sec == pytest.approx(0.4)
    assert env._manager.close_settle_sec == pytest.approx(0.2)
    assert env._manager.process_tree_wait_sec == pytest.approx(0.3)
    assert env._conf_float("sc2_retry_backoff_sec", 2.0) == pytest.approx(1.5)
    # Timeout budget includes worker_index × pipeline stagger + join-slack + in_game.
    assert env._startup_timeout_sec() == pytest.approx(90.0 + 15.0 + 3.75 + 24.0)


def test_single_client_env_reset_raises_after_max_startup_retries(mock_sc2_process, mock_client):
    env = SingleClientEnv(
        sc2_process_cls=lambda **kwargs: mock_sc2_process,
        client_cls=lambda **kwargs: mock_client,
        training_conf={"env": {"startup_max_retries": 2}},
    )
    env._setup_game_session = AsyncMock(side_effect=RuntimeError("SC2 won't start"))
    env._manager.launch_process = AsyncMock()
    env.close = MagicMock()

    with patch("train.envs.single_client_env.time.sleep"):
        with pytest.raises(RuntimeError, match="SC2 won't start"):
            env.reset()

    assert env._manager.launch_process.call_count == 2
    assert env.close.call_count == 2


def test_single_client_env_step_survives_connection_already_closed():
    from sc2.protocol import ConnectionAlreadyClosed

    env = SingleClientEnv(training_conf={})
    env.game_ended = False
    env._last_obs = {"map": np.zeros((1, 2, 2), dtype=np.float32)}
    env._loop = MagicMock()
    env._loop.run_until_complete = MagicMock(
        side_effect=ConnectionAlreadyClosed("Connection already closed.")
    )
    env.close = MagicMock()
    env._log = MagicMock()

    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated is True
    assert reward == 0.0
    assert info.get("sc2_connection_lost") is True
    env.close.assert_called_once()

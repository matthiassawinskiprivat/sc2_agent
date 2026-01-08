import pytest
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from unittest.mock import MagicMock

def test_ctf_env_pipeline_instantiation():
    """
    Test that CaptureTheFlagEnv can be instantiated with the standard arguments
    passed by the PipelineManager.
    """
    level_name = "capture_the_flag"
    headless = True
    grid_size = 128
    rank = 0
    num_envs = 1
    sc2_process_cls = MagicMock()
    client_cls = MagicMock()
    combined_conf = {
        "training_steps": 1000,
        "batch_size": 128,
        "grid_size": 128
    }
    max_steps = 300

    try:
        env = CaptureTheFlagEnv(
            level_name=level_name,
            visualize=not headless,
            grid_size=grid_size,
            worker_index=rank,
            num_workers=num_envs,
            sc2_process_cls=sc2_process_cls,
            client_cls=client_cls,
            training_conf=combined_conf,
            max_steps=max_steps
        )
        assert env is not None
        assert env.level_name == level_name
        assert env._sc2_process_cls == sc2_process_cls
        assert env._client_cls == client_cls
    except TypeError as e:
        pytest.fail(f"CaptureTheFlagEnv instantiation failed with TypeError: {e}")

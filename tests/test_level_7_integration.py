import pytest

def test_level_7_env_exported():
    try:
        from train.envs import Level7Env
        assert Level7Env is not None
    except ImportError:
        pytest.fail("Level7Env not exported from train.envs")

def test_level_7_env_registered():
    from train.train_pipeline import ENV_REGISTRY
    assert "Level7Env" in ENV_REGISTRY
    from train.envs.level_7_env import Level7Env
    assert ENV_REGISTRY["Level7Env"] == Level7Env

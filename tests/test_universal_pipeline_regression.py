"""
Pipeline regression gate — mandatory CI step.

Verifies that every non-CTF environment already integrated with the
universal pipeline is UNCHANGED after any CTF-related work.

This file must NOT be edited as part of CTF work.  Any edit here
requires a separate PR with a justification and sign-off from the
pipeline owner.
"""
from __future__ import annotations

import pytest
import numpy as np
from typing import Any, Dict

from src.universal.env_registry import EnvRegistry

NON_CTF_ENV_IDS: list[str] = [
    # Add non-CTF universal env IDs here as they are integrated.
    # Example:
    # "grid_nav_v1",
]

EXPECTED_OBS_SHAPES: Dict[str, tuple] = {}
DEFAULT_CONFIGS: Dict[str, Any] = {}


def default_config(env_id: str) -> Any:
    if env_id in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[env_id]
    raise KeyError(
        f"No default config registered for '{env_id}'.  "
        "Add it to DEFAULT_CONFIGS in test_universal_pipeline_regression.py."
    )


@pytest.mark.parametrize("env_id", NON_CTF_ENV_IDS)
def test_env_unchanged(env_id: str) -> None:
    env = EnvRegistry.make(env_id, default_config(env_id))
    try:
        obs = env.reset()
        if env_id in EXPECTED_OBS_SHAPES:
            actual = env.observation_space.shape
            assert actual == EXPECTED_OBS_SHAPES[env_id]

        action = env.action_space.sample()
        obs2, rewards, done, info = env.step(
            {"__default__": action} if isinstance(action, np.ndarray) else action
        )

        if env_id in EXPECTED_OBS_SHAPES and "spatial_global" in obs2:
            assert obs2["spatial_global"].shape == obs["spatial_global"].shape

        assert isinstance(done, bool)
        if isinstance(rewards, dict):
            for v in rewards.values():
                assert isinstance(v, (int, float))
        else:
            assert isinstance(rewards, (int, float))

    finally:
        env.close()


def test_ctf_env_registered() -> None:
    """Importing train.envs.ctf registers ctf_sc2_v1 in EnvRegistry."""
    import train.envs.ctf  # noqa: F401
    assert "ctf_sc2_v1" in EnvRegistry.list_ids()


def test_non_ctf_envs_still_registered_after_ctf_import() -> None:
    import train.envs.ctf  # noqa: F401
    registered = EnvRegistry.list_ids()
    for env_id in NON_CTF_ENV_IDS:
        assert env_id in registered

"""
Universal base environment contract.

All training environments in the pipeline must subclass UniversalBaseEnv
and implement the abstract methods.  No behaviour of this module may be
changed in a way that alters existing subclass behaviour without a
corresponding update to tests/test_universal_pipeline_regression.py.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ActionSpace(abc.ABC):
    """Minimal action-space descriptor."""

    @abc.abstractmethod
    def sample(self) -> Any:
        """Return one random valid action."""

    @abc.abstractmethod
    def contains(self, action: Any) -> bool:
        """Return True if *action* is a valid member of this space."""


class BoxActionSpace(ActionSpace):
    """Continuous box action space backed by numpy."""

    def __init__(self, low: np.ndarray, high: np.ndarray, dtype: np.dtype = np.float32) -> None:
        assert low.shape == high.shape, "low and high must have the same shape"
        self.low = low.astype(dtype)
        self.high = high.astype(dtype)
        self.dtype = dtype
        self.shape: Tuple[int, ...] = low.shape

    def sample(self) -> np.ndarray:
        return (np.random.random(self.shape) * (self.high - self.low) + self.low).astype(self.dtype)

    def contains(self, action: Any) -> bool:
        a = np.asarray(action)
        return bool(a.shape == self.shape and np.all(a >= self.low) and np.all(a <= self.high))


class ObservationSpace(abc.ABC):
    """Minimal observation-space descriptor."""

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of a single flat observation array."""


class BoxObservationSpace(ObservationSpace):
    """Observation space described by a numpy array shape and bounds."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        low: float = 0.0,
        high: float = 1.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        self._shape = shape
        self.low = low
        self.high = high
        self.dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class UniversalBaseEnv(abc.ABC):
    """
    Abstract base class for every environment in the universal training
    pipeline.

    Contract
    --------
    * ``reset(**kwargs) -> Dict[str, Any]``
        Reset to initial state; return the first observation dict.
    * ``step(actions) -> (obs, rewards, done, info)``
        Advance one environment step.  *actions* maps squad_id → action
        tensor.  *rewards* maps squad_id → scalar float.
    * ``close()``
        Release SC2 process / network connections / file handles.
    * ``action_space`` property → ActionSpace
    * ``observation_space`` property → ObservationSpace

    Optional hooks (default no-ops): ``seed``, ``render``,
    ``get_obs_version``.
    """

    @abc.abstractmethod
    def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """Reset environment state; return initial observation."""

    @abc.abstractmethod
    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Advance the environment by one step."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all external resources."""

    @property
    @abc.abstractmethod
    def action_space(self) -> ActionSpace:
        """Action-space descriptor for a single squad step."""

    @property
    @abc.abstractmethod
    def observation_space(self) -> ObservationSpace:
        """Observation-space descriptor for a single squad step."""

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the internal RNG.  Override for reproducibility."""

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render current state.  Override for visualisation."""
        return None

    def get_obs_version(self) -> str:
        """Return the feature-version string embedded in each obs info."""
        return "base_v1"

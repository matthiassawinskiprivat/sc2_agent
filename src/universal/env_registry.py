"""
Central environment registry.

Environments self-register on import via ``EnvRegistry.register()``.
The universal trainer calls ``EnvRegistry.make(env_id, config)`` to
instantiate a concrete env without knowing its class directly.

Non-CTF environments already registered here must not be altered by
CTF-specific changes; they are protected by tests/test_universal_pipeline_regression.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Type

from src.universal.base_env import UniversalBaseEnv


class EnvRegistry:
    """
    Singleton registry mapping *env_id* strings to env classes.

    Usage
    -----
    >>> EnvRegistry.register("my_env_v1", MyEnv)
    >>> env = EnvRegistry.make("my_env_v1", config)
    """

    _registry: Dict[str, Type[UniversalBaseEnv]] = {}

    @classmethod
    def register(cls, env_id: str, env_class: Type[UniversalBaseEnv]) -> None:
        """
        Register *env_class* under *env_id*.

        Raises ``ValueError`` if *env_id* is already taken, preventing
        silent shadowing of existing environments.
        """
        if env_id in cls._registry:
            raise ValueError(
                f"Environment '{env_id}' is already registered.  "
                "Use a different env_id or call deregister() first."
            )
        if not (isinstance(env_class, type) and issubclass(env_class, UniversalBaseEnv)):
            raise TypeError(
                f"env_class must be a subclass of UniversalBaseEnv, got {env_class!r}"
            )
        cls._registry[env_id] = env_class

    @classmethod
    def make(cls, env_id: str, config: Any) -> UniversalBaseEnv:
        """
        Instantiate the environment registered under *env_id*.

        Raises ``KeyError`` if *env_id* is not registered.
        """
        if env_id not in cls._registry:
            available = sorted(cls._registry)
            raise KeyError(
                f"Environment '{env_id}' is not registered.  "
                f"Available: {available}"
            )
        return cls._registry[env_id](config)

    @classmethod
    def list_ids(cls) -> List[str]:
        """Return sorted list of all registered env IDs."""
        return sorted(cls._registry.keys())

    @classmethod
    def deregister(cls, env_id: str) -> None:
        """Remove *env_id* from the registry.  Safe to call if not present."""
        cls._registry.pop(env_id, None)

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Wipe the registry.  For use in isolated test suites only."""
        cls._registry.clear()

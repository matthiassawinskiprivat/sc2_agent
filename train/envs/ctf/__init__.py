"""
CTF SC2 environment package (universal pipeline).

Importing this module registers ``ctf_sc2_v1`` in the universal
``EnvRegistry``.  Any module that needs to instantiate the CTF env
must first import this package (directly or transitively).

The registration is guarded against optional dependencies (pydantic,
EnvRegistry) so that lightweight imports of sub-modules (e.g.
``region_loader``) do not fail in environments where those
dependencies are absent.
"""
try:
    from train.envs.ctf.env import CTFEnv
    from src.universal.env_registry import EnvRegistry
    EnvRegistry.register(CTFEnv.ENV_ID, CTFEnv)
    __all__ = ["CTFEnv"]
except Exception:
    __all__ = []

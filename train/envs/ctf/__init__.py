"""
CTF SC2 environment package (universal pipeline).

Importing this module registers ``ctf_sc2_v1`` in the universal
``EnvRegistry``.  Any module that needs to instantiate the CTF env
must first import this package (directly or transitively).
"""
from train.envs.ctf.env import CTFEnv
from src.universal.env_registry import EnvRegistry

EnvRegistry.register(CTFEnv.ENV_ID, CTFEnv)

__all__ = ["CTFEnv"]

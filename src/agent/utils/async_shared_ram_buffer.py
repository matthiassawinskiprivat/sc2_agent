"""
Async rollout buffer IPC via ``multiprocessing.shared_memory`` (RAM, no observation memmap).

Collector and learner are separate processes; large map/tactical tensors are stored in
named shared segments. Only small arrays + signals use disk (payload.npz + JSON).
"""
from __future__ import annotations

import math
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from src.agent.utils.async_disk_buffer import AsyncDiskBuffer


def _shared_observation_keys(observation_space) -> List[str]:
    return [key for key in observation_space.spaces if key in ("map", "tactical")]


def _unlink_shm_by_name(name: str) -> None:
    try:
        existing = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    try:
        existing.close()
        existing.unlink()
    except Exception:
        pass


def observation_buffer_nbytes(shape) -> int:
    """Byte size for a float32 observation tensor (avoids np.prod int32 overflow on CTF)."""
    nbytes = math.prod(int(x) for x in shape) * np.dtype(np.float32).itemsize
    if nbytes <= 0:
        raise ValueError(f"Shared memory size must be positive for shape {shape}")
    return int(nbytes)


def _create_shm_segment(name: str, nbytes: int) -> SharedMemory:
    nbytes = int(nbytes)
    if nbytes <= 0:
        raise ValueError(f"Shared memory size must be positive for segment {name!r}")
    _unlink_shm_by_name(name)
    return SharedMemory(name=name, create=True, size=nbytes)


class AsyncSharedRamBuffer(AsyncDiskBuffer):
    """
    Cross-process async buffer: map/tactical in shared RAM; signaling unchanged.

    RNN minibatch sequencing (``episode_starts`` / focal-unit splits) is inherited from
    ``DiskDictRolloutBuffer.get`` — CTF async training uses this class, not a separate path.
    """

    def __init__(self, *args, **kwargs):
        kwargs["storage_backend"] = "ram"
        super().__init__(*args, **kwargs)
        self._shm_handles: Dict[str, SharedMemory] = {}

    def shared_observation_keys(self) -> List[str]:
        return _shared_observation_keys(self.observation_space)

    def shm_segment_name(self, key: str) -> str:
        return f"sc2async_{self.instance_id}_{key}"

    def reset(self) -> None:
        """
        Reset rollout scalars but keep map/tactical views backed by shared memory.

        ``DictRolloutBuffer.reset()`` replaces every observation array with fresh
        ``np.zeros`` in process-local RAM. For CTF async IPC that silently disconnects
        the collector from its shm segments: forwards/log_probs use real obs while the
        learner reads zeros from shm (approx_kl ~7 despite rollout_policy_load_ok=1).
        """
        shm_obs = {}
        handles = getattr(self, "_shm_handles", None)
        if handles and hasattr(self, "observations"):
            for key in self.shared_observation_keys():
                if key in handles and key in self.observations:
                    shm_obs[key] = self.observations[key]
        super().reset()
        for key, arr in shm_obs.items():
            self.observations[key] = arr
        if hasattr(self, "policy_active"):
            self.policy_active.fill(0.0)

    def bind_shared_observation_buffers(self) -> None:
        """Create (collector) or attach (learner) shared segments for large obs tensors."""
        self.close_shared_memory(unlink=False)
        for key in self.shared_observation_keys():
            subspace = self.observation_space.spaces[key]
            shape = (self.buffer_size, self.n_envs, *subspace.shape)
            nbytes = observation_buffer_nbytes(shape)
            name = self.shm_segment_name(key)
            shm = _create_shm_segment(name, nbytes)
            self._shm_handles[key] = shm
            self.observations[key] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        logger.debug(
            f"Shared RAM obs bound for {self.instance_id}: "
            f"{list(self._shm_handles.keys())}"
        )

    @staticmethod
    def attach_shared_observations(buffer: "AsyncSharedRamBuffer", meta: Dict[str, Any]) -> None:
        """Learner: map existing collector segments by name from signal metadata."""
        shared_obs = meta.get("shared_obs") or {}
        handles: Dict[str, SharedMemory] = getattr(buffer, "_shm_handles", {})
        for key, info in shared_obs.items():
            name = info["name"]
            shape = tuple(info["shape"])
            if key in handles:
                try:
                    handles[key].close()
                except Exception:
                    pass
            shm = SharedMemory(name=name, create=False)
            handles[key] = shm
            buffer.observations[key] = np.ndarray(
                shape, dtype=np.float32, buffer=shm.buf
            )
        buffer._shm_handles = handles

    def close_shared_memory(self, *, unlink: bool = False) -> None:
        for shm in list(self._shm_handles.values()):
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except Exception:
                pass
        self._shm_handles.clear()

    def signal_ready(self, metadata: Dict[str, Any]):
        import json
        import os

        payload_path = os.path.join(self.instance_dir, "payload.npz")
        shared_keys = set(self.shared_observation_keys())
        ram_observations = {}
        for key, obs in self.observations.items():
            if key in shared_keys:
                continue
            if not isinstance(obs, np.memmap):
                ram_observations[f"obs_{key}"] = obs

        tmp_payload = os.path.join(self.instance_dir, "payload_staging.npz")
        np.savez(
            tmp_payload,
            actions=self.actions,
            rewards=self.rewards,
            episode_starts=self.episode_starts,
            values=self.values,
            log_probs=self.log_probs,
            advantages=self.advantages,
            returns=self.returns,
            hidden_states_pi=self.hidden_states_pi,
            hidden_states_vf=self.hidden_states_vf,
            cell_states_pi=getattr(self, "cell_states_pi", np.array([])),
            cell_states_vf=getattr(self, "cell_states_vf", np.array([])),
            action_masks=getattr(self, "action_masks", np.array([])),
            policy_active=getattr(self, "policy_active", np.array([])),
            **ram_observations,
        )
        os.replace(tmp_payload, payload_path)

        metadata["buffer_size"] = self.buffer_size
        metadata["n_envs"] = self.n_envs
        metadata["observation_space"] = {
            k: v.shape for k, v in self.observation_space.spaces.items()
        }
        metadata["ipc_backend"] = "shared_memory"
        metadata["shared_obs"] = {
            key: {
                "name": self.shm_segment_name(key),
                "shape": list(self.observations[key].shape),
            }
            for key in self.shared_observation_keys()
        }

        tmp_signal = self.signal_file + ".tmp"
        with open(tmp_signal, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp_signal, self.signal_file)
        logger.debug(f"Buffer signaled ready (shared RAM IPC): {self.instance_id}")

    def __del__(self):
        try:
            self.close_shared_memory(unlink=True)
        except Exception:
            pass

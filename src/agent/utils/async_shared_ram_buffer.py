"""
Async rollout buffer IPC via ``multiprocessing.shared_memory`` (RAM, no observation memmap).

Collector and learner are separate processes; large map/tactical tensors are stored in
named shared segments. Only small arrays + signals use disk (payload.npz + JSON).
"""
from __future__ import annotations

import ctypes
import math
import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.agent.utils.disk_buffer import (
    assert_obs_fingerprint_matches,
    is_disk_offloaded_obs_key,
    obs_fingerprint_payload,
)
from src.agent.utils.async_disk_buffer import AsyncDiskBuffer


def _shared_observation_keys(observation_space) -> List[str]:
    return [key for key in observation_space.spaces if is_disk_offloaded_obs_key(key)]


class _WinApiMappedSegment:
    """Open an existing Windows named mapping (never CreateFileMapping).

    ``mmap.mmap(-1, size, tagname=...)`` can create a **new empty** segment when
    the collector name is not found, so the learner trains on zeros (explained
    variance collapses). ``OpenFileMapping`` + ``MapViewOfFile`` can only attach.
    """

    def __init__(self, name: str, h_map: int, ptr: int, nbytes: int):
        self.name = name
        self._h_map = h_map
        self._ptr = ptr
        self._size = int(nbytes)
        self._ctypes_buf = (ctypes.c_byte * self._size).from_address(int(ptr))
        self._buf: Optional[memoryview] = memoryview(self._ctypes_buf)

    @property
    def buf(self):
        return self._buf

    @property
    def size(self) -> int:
        return self._size

    def close(self) -> None:
        import _winapi

        if self._buf is not None:
            try:
                self._buf.release()
            except Exception:
                pass
            self._buf = None
        self._ctypes_buf = None
        if self._ptr:
            try:
                _winapi.UnmapViewOfFile(self._ptr)
            except Exception:
                pass
            self._ptr = None
        if self._h_map:
            try:
                _winapi.CloseHandle(self._h_map)
            except Exception:
                pass
            self._h_map = None

    def unlink(self) -> None:
        return


def _attach_shm_windows(name: str, nbytes: int) -> _WinApiMappedSegment:
    import _winapi

    nbytes = int(nbytes)
    h_map = None
    access = _winapi.FILE_MAP_WRITE
    try:
        h_map = _winapi.OpenFileMapping(access, False, name)
    except OSError as exc:
        try:
            h_map = _winapi.OpenFileMapping(_winapi.FILE_MAP_READ, False, name)
            access = _winapi.FILE_MAP_READ
        except OSError as exc2:
            raise FileNotFoundError(name) from exc2
    if not h_map:
        raise FileNotFoundError(name)
    try:
        ptr = _winapi.MapViewOfFile(h_map, access, 0, 0, nbytes)
    except OSError:
        _winapi.CloseHandle(h_map)
        raise
    if not ptr:
        _winapi.CloseHandle(h_map)
        raise OSError(f"MapViewOfFile failed for {name!r} ({nbytes} bytes)")
    return _WinApiMappedSegment(name, h_map, ptr, nbytes)


def _attach_shm_segment(name: str, nbytes: int) -> Any:
    """Open an existing named segment without CPython's Windows double-map."""
    nbytes = int(nbytes)
    if nbytes <= 0:
        raise ValueError(f"Shared memory size must be positive for segment {name!r}")
    if os.name == "nt":
        return _attach_shm_windows(name, nbytes)
    return SharedMemory(name=name, create=False)


def _unlink_shm_by_name(name: str) -> None:
    try:
        existing = SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    except OSError:
        # Huge leftover segments: Windows attach can raise WinError 1450.
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
        if hasattr(self, "rnn_context_ids"):
            self.rnn_context_ids.fill(-1)

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
        total = sum(h.size if hasattr(h, "size") else len(h.buf) for h in self._shm_handles.values())
        logger.info(
            f"Shared RAM obs bound for {self.instance_id}: "
            f"{list(self._shm_handles.keys())} ({total / 1e9:.2f} GB)"
        )

    @staticmethod
    def attach_shared_observations(buffer: "AsyncSharedRamBuffer", meta: Dict[str, Any]) -> None:
        """Learner: map existing collector segments by name from signal metadata."""
        shared_obs = meta.get("shared_obs") or {}
        handles: Dict[str, Any] = getattr(buffer, "_shm_handles", {})
        for key, info in shared_obs.items():
            name = info["name"]
            shape = tuple(info["shape"])
            nbytes = int(info.get("nbytes") or observation_buffer_nbytes(shape))
            if key in handles:
                try:
                    handles[key].close()
                except Exception:
                    pass
            try:
                shm = _attach_shm_segment(name, nbytes)
            except OSError as exc:
                gb = nbytes / 1e9
                logger.error(
                    f"Failed to attach shared obs {key!r} ({name}, {nbytes} bytes, "
                    f"{gb:.2f} GB). WinError 1450 means Windows could not map the "
                    f"CTF RAM buffer; try rollout_buffer: disk or lower n_steps/"
                    f"num_envs. Original: {exc}"
                )
                raise
            handles[key] = shm
            mapped = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
            assert_obs_fingerprint_matches(key, mapped, info, kind="Shared")
            buffer.observations[key] = mapped
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
            rnn_context_ids=getattr(self, "rnn_context_ids", np.array([])),
            **ram_observations,
        )
        os.replace(tmp_payload, payload_path)

        metadata["buffer_size"] = self.buffer_size
        metadata["n_envs"] = self.n_envs
        metadata["observation_space"] = {
            k: v.shape for k, v in self.observation_space.spaces.items()
        }
        metadata["ipc_backend"] = "shared_memory"
        metadata["shared_obs"] = {}
        for key in self.shared_observation_keys():
            obs = self.observations[key]
            metadata["shared_obs"][key] = {
                "name": self.shm_segment_name(key),
                "nbytes": observation_buffer_nbytes(obs.shape),
                **obs_fingerprint_payload(obs),
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

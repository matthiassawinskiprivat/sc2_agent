"""Shared-memory opponent peek for squad-pointer collector batching.

Workers write ``map`` / ``anchor_tactical`` into collector-owned named segments
so ``peek_opponent_squad_pointer`` does not pickle those tensors through the
Subproc pipe. Attach uses the same Windows-safe open as the rollout buffer
(``OpenFileMapping``, never ``mmap(-1, …)`` which can create an empty segment).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.agent.utils.async_shared_ram_buffer import (
    _attach_shm_segment,
    _create_shm_segment,
    observation_buffer_nbytes,
)

PEEK_SHM_KEYS = ("map", "anchor_tactical")


def spatial_sample_mean(arr: np.ndarray) -> float:
    """Fingerprint one spatial tensor (first channel mean)."""
    if arr is None or getattr(arr, "size", 0) == 0:
        return 0.0
    if arr.ndim >= 3:
        return float(np.mean(arr[0]))
    return float(np.mean(arr))


def peek_shm_keys_from_space(observation_space) -> List[str]:
    spaces = getattr(observation_space, "spaces", None)
    if not isinstance(spaces, dict):
        return []
    return [key for key in PEEK_SHM_KEYS if key in spaces]


class OpponentPeekSharedMemory:
    """Collector-owned ``(n_envs, *shape)`` views for opponent spatial tensors."""

    def __init__(self, num_envs: int, observation_space, *, instance_id: Optional[str] = None):
        self.num_envs = int(num_envs)
        self.instance_id = instance_id or f"opppeek_{os.getpid()}_{id(self)}"
        self._handles: Dict[str, Any] = {}
        self.views: Dict[str, np.ndarray] = {}
        self._meta_keys: Dict[str, Dict[str, Any]] = {}
        for key in peek_shm_keys_from_space(observation_space):
            subspace = observation_space.spaces[key]
            item_shape = tuple(int(x) for x in subspace.shape)
            shape = (self.num_envs, *item_shape)
            nbytes = observation_buffer_nbytes(shape)
            name = f"sc2{self.instance_id}_{key}"
            shm = _create_shm_segment(name, nbytes)
            self._handles[key] = shm
            view = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
            view.fill(0.0)
            self.views[key] = view
            self._meta_keys[key] = {
                "name": shm.name,
                "shape": list(shape),
                "item_shape": list(item_shape),
                "nbytes": nbytes,
            }

    def worker_meta(self) -> Dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "keys": {key: dict(info) for key, info in self._meta_keys.items()},
        }

    def view(self, key: str, slot: int) -> Optional[np.ndarray]:
        arr = self.views.get(key)
        if arr is None:
            return None
        idx = int(slot)
        if idx < 0 or idx >= arr.shape[0]:
            return None
        return arr[idx]

    def close(self, *, unlink: bool = True) -> None:
        for shm in list(self._handles.values()):
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except Exception:
                pass
        self._handles.clear()
        self.views.clear()


def attach_peek_views(meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Worker: map collector segments (write) without creating new names."""
    keys = (meta or {}).get("keys") or {}
    views: Dict[str, np.ndarray] = {}
    for key, info in keys.items():
        name = info["name"]
        shape = tuple(int(x) for x in info["shape"])
        nbytes = int(info.get("nbytes") or observation_buffer_nbytes(shape))
        shm = _attach_shm_segment(name, nbytes)
        views[key] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        views[f"_{key}_handle"] = shm  # keep mapped for process lifetime
    return views


def write_peek_spatial(
    views: Dict[str, np.ndarray],
    slot: int,
    obs: Dict[str, np.ndarray],
) -> Optional[Dict[str, float]]:
    """Copy large spatial keys into ``views[key][slot]``. None if shapes disagree."""
    idx = int(slot)
    means: Dict[str, float] = {}
    for key in PEEK_SHM_KEYS:
        dest_full = views.get(key)
        src = obs.get(key)
        if dest_full is None or src is None:
            continue
        if idx < 0 or idx >= dest_full.shape[0]:
            return None
        dest = dest_full[idx]
        arr = np.asarray(src, dtype=np.float32)
        if arr.shape != dest.shape:
            return None
        np.copyto(dest, arr)
        means[key] = spatial_sample_mean(dest)
    return means


def compact_step_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Keep space keys so VecEnv can stack, but do not pickle the full maps."""
    out = dict(obs)
    for key in PEEK_SHM_KEYS:
        if key in out:
            out[key] = np.zeros((1,), dtype=np.float32)
    return out


def _info_row(infos: Any, index: int) -> Optional[dict]:
    if infos is None:
        return None
    if isinstance(infos, dict):
        return infos if index == 0 else None
    try:
        row = infos[index]
    except (TypeError, IndexError, KeyError):
        return None
    return row if isinstance(row, dict) else None


def hydrate_vec_spatial_obs(
    obs: Any,
    infos: Any,
    shm: Optional[OpponentPeekSharedMemory],
    num_envs: int,
    *,
    reset_infos: Any = None,
    dones: Any = None,
) -> Any:
    """Fill batched ``map`` / ``anchor_tactical`` from shared slots after ``step``/``reset``.

    On episode ``done`` the observation is the *reset* obs (already in shm) while
    ``infos`` still describes the terminal step — use ``reset_infos`` then.
    """
    if shm is None or not isinstance(obs, dict) or num_envs <= 0:
        return obs
    done_row = None
    if dones is not None:
        done_row = np.asarray(dones, dtype=bool).reshape(-1)
    keys = [key for key in PEEK_SHM_KEYS if key in shm.views]
    if not keys:
        return obs
    stacked: Dict[str, List[np.ndarray]] = {key: [] for key in keys}
    for i in range(int(num_envs)):
        info = None
        if done_row is not None and i < done_row.shape[0] and done_row[i]:
            info = _info_row(reset_infos, i)
        if info is None or not info.get("spatial_shm"):
            info = _info_row(infos, i)
        if info is None or not info.get("spatial_shm"):
            return obs
        slot = int(info.get("spatial_shm_slot", i))
        means = info.get("spatial_shm_means") if isinstance(info.get("spatial_shm_means"), dict) else {}
        for key in keys:
            view = shm.view(key, slot)
            if view is None:
                return obs
            expected = means.get(key)
            if expected is not None:
                got = spatial_sample_mean(view)
                tol = max(1e-4, 1e-3 * abs(float(expected)))
                if not np.isfinite(got) or abs(got - float(expected)) > tol:
                    return obs
            stacked[key].append(np.array(view, copy=True))
    out = dict(obs)
    for key, rows in stacked.items():
        out[key] = np.stack(rows, axis=0)
    return out


def compact_opponent_peek_obs(
    obs: Dict[str, np.ndarray],
    *,
    slot: int,
    sample_means: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """Copy obs without spatial keys (cached pool obs must stay intact)."""
    out = {
        key: value
        for key, value in obs.items()
        if key not in PEEK_SHM_KEYS
    }
    out["_peek_shm"] = True
    out["_peek_slot"] = int(slot)
    out["_peek_means"] = dict(sample_means)
    return out


def reconstruct_opponent_peek_obs(
    payload: Dict[str, Any],
    shm: Optional[OpponentPeekSharedMemory],
    env_idx: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Rebuild a full opponent obs from compact peek + shared spatial slots."""
    obs = payload.get("obs") if isinstance(payload, dict) else None
    if not isinstance(obs, dict):
        return None
    if not obs.get("_peek_shm"):
        return {
            key: value
            for key, value in obs.items()
            if not str(key).startswith("_peek")
        }
    if shm is None:
        return None
    slot = int(obs.get("_peek_slot", env_idx))
    means = obs.get("_peek_means") if isinstance(obs.get("_peek_means"), dict) else {}
    out = {
        key: value
        for key, value in obs.items()
        if key not in PEEK_SHM_KEYS and not str(key).startswith("_peek")
    }
    for key in PEEK_SHM_KEYS:
        view = shm.view(key, slot)
        if view is None:
            if key in means:
                return None
            continue
        expected = means.get(key)
        if expected is not None:
            got = spatial_sample_mean(view)
            tol = max(1e-4, 1e-3 * abs(float(expected)))
            if not np.isfinite(got) or abs(got - float(expected)) > tol:
                return None
        out[key] = np.array(view, copy=True)
    return out


def attach_opponent_peek_to_vec(vec_env, shm: OpponentPeekSharedMemory) -> None:
    if shm is None or not hasattr(vec_env, "env_method"):
        return
    try:
        if hasattr(vec_env, "has_attr") and not vec_env.has_attr(
            "attach_opponent_peek_shm"
        ):
            return
        vec_env.env_method("attach_opponent_peek_shm", shm.worker_meta())
    except Exception:
        return


def pack_pending_opponent_actions(
    num_envs: int,
    ready: Iterable[tuple],
    actions: np.ndarray,
    new_lstms: List[Any],
) -> List[Optional[tuple]]:
    packed: List[Optional[tuple]] = [None] * int(num_envs)
    for j, (env_idx, payload) in enumerate(ready):
        action = np.asarray(actions[j], dtype=np.int64).reshape(-1)
        lstm = new_lstms[j] if j < len(new_lstms) else None
        packed[int(env_idx)] = (action, lstm, payload.get("pool_idx"))
    return packed

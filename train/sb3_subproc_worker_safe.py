"""
Safe SB3 SubprocVecEnv worker for Windows / curriculum teardown.

When the parent tears down SubprocVecEnv (e.g. between pipeline levels), workers can
still be inside remote.send() after env.reset() following a terminal step. If the
parent has already closed its end of the pipe, workers raise BrokenPipeError and
spam tracebacks. SB3's stock _worker does not catch this.

Also retries env.reset() after terminal steps so a single SC2 crash does not kill
the worker process (which would take down the whole SubprocVecEnv).
"""
import multiprocessing as mp
import os
import time
from typing import Any, Optional, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _send_safe(remote: mp.connection.Connection, payload: Any) -> bool:
    try:
        remote.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _zero_observation(env) -> Any:
    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Dict):
        return {
            k: np.zeros(v.shape, dtype=v.dtype)
            for k, v in obs_space.spaces.items()
        }
    return np.zeros(obs_space.shape, dtype=obs_space.dtype)


def _reset_env_with_retry(env, *, seed=None, options=None) -> Tuple[Any, dict]:
    """Reset with backoff; survive SC2 relaunch failures without killing the worker."""
    max_retries = int(os.environ.get("SC2_WORKER_RESET_RETRIES", "3"))
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            if seed is not None or options:
                maybe_options = {"options": options} if options else {}
                return env.reset(seed=seed, **maybe_options)
            return env.reset()
        except Exception as exc:
            last_exc = exc
            try:
                env.close()
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(8.0 + attempt * 2.0)

    print(
        f"[SubprocWorker] env.reset() failed after {max_retries} attempts "
        f"({type(last_exc).__name__}: {last_exc}); returning zero observation so "
        "collector can rebuild vec env."
    )
    return _zero_observation(env), {"sc2_reset_failed": True}


def _safe_worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, BrokenPipeError, OSError):
            break
        except KeyboardInterrupt:
            break

        if cmd == "step":
            observation, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                info["terminal_observation"] = observation
                observation, reset_info = _reset_env_with_retry(env)
            if not _send_safe(remote, (observation, reward, done, info, reset_info)):
                break
        elif cmd == "reset":
            observation, reset_info = _reset_env_with_retry(
                env, seed=data[0], options=data[1] if data[1] else None
            )
            if not _send_safe(remote, (observation, reset_info)):
                break
        elif cmd == "render":
            if not _send_safe(remote, env.render()):
                break
        elif cmd == "close":
            env.close()
            try:
                remote.close()
            except Exception:
                pass
            break
        elif cmd == "get_spaces":
            if not _send_safe(remote, (env.observation_space, env.action_space)):
                break
        elif cmd == "env_method":
            method = env.get_wrapper_attr(data[0])
            if not _send_safe(remote, method(*data[1], **data[2])):
                break
        elif cmd == "get_attr":
            if not _send_safe(remote, env.get_wrapper_attr(data)):
                break
        elif cmd == "has_attr":
            try:
                env.get_wrapper_attr(data)
                if not _send_safe(remote, True):
                    break
            except AttributeError:
                if not _send_safe(remote, False):
                    break
        elif cmd == "set_attr":
            if not _send_safe(remote, setattr(env, data[0], data[1])):  # type: ignore[func-returns-value]
                break
        elif cmd == "is_wrapped":
            if not _send_safe(remote, is_wrapped(env, data)):
                break
        else:
            raise NotImplementedError(f"`{cmd}` is not implemented in the worker")


def apply_sb3_subproc_worker_patch() -> None:
    import stable_baselines3.common.vec_env.subproc_vec_env as smv

    smv._worker = _safe_worker

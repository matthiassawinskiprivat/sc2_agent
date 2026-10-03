import os
import numpy as np
import torch as th
import json
import time
from typing import Dict, Any, Optional, Union, Tuple
from gymnasium import spaces
from loguru import logger
from src.agent.utils.disk_buffer import (
    DiskDictRolloutBuffer,
    assert_obs_fingerprint_matches,
    is_disk_offloaded_obs_key,
    obs_fingerprint_payload,
)

class AsyncDiskBuffer(DiskDictRolloutBuffer):
    """
    An extension of DiskDictRolloutBuffer that supports cross-process 
    signaling for Asynchronous Producer-Consumer training.
    """
    @property
    def signal_file(self):
        return os.path.join(self.instance_dir, "buffer_ready.json")

    @property
    def lock_file(self):
        return os.path.join(self.instance_dir, "buffer.lock")

    def signal_ready(self, metadata: Dict[str, Any]):
        """
        Actor calls this to tell the Learner: 'I've finished filling this buffer!'
        """
        for obs in self.observations.values():
            if isinstance(obs, np.memmap):
                obs.flush()

        # Save RAM-based payload to disk for the Learner
        payload_path = os.path.join(self.instance_dir, "payload.npz")
        
        # Collect all observations that are NOT in memmaps
        ram_observations = {}
        for key, obs in self.observations.items():
            if not isinstance(obs, np.memmap):
                ram_observations[f"obs_{key}"] = obs

        # savez (uncompressed) is used here deliberately.  savez_compressed reduces file
        # size but the zlib compression/decompression of large float32 rollout arrays
        # (hidden states, observations, values…) costs more CPU time per rollout than the
        # disk-bandwidth savings are worth — especially on SSDs.  The atomic rename below
        # still protects against partial-read races between the Collector and Learner.
        # Temp name must end with `.npz` — NumPy appends `.npz` if the path does not, so
        # `payload.npz.tmp` becomes `payload.npz.tmp.npz` and os.replace fails on Windows.
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
            cell_states_pi=getattr(self, 'cell_states_pi', np.array([])),
            cell_states_vf=getattr(self, 'cell_states_vf', np.array([])),
            action_masks=getattr(self, 'action_masks', np.array([])),
            policy_active=getattr(self, 'policy_active', np.array([])),
            **ram_observations
        )
        os.replace(tmp_payload, payload_path)

        # Save structural metadata so the Learner knows exactly how to read the binary file
        metadata["buffer_size"] = self.buffer_size
        metadata["n_envs"] = self.n_envs
        metadata["observation_space"] = {k: v.shape for k, v in self.observation_space.spaces.items()}
        metadata["ipc_backend"] = "disk"
        # Fingerprints for memmap obs — learner must reopen the same bytes (Windows
        # can otherwise serve a stale/empty view → multi-nat logπ mismatch).
        disk_obs = {}
        for key, obs in self.observations.items():
            if is_disk_offloaded_obs_key(key) and isinstance(obs, np.memmap):
                disk_obs[key] = obs_fingerprint_payload(obs)
        if disk_obs:
            metadata["disk_obs"] = disk_obs

        tmp_signal = self.signal_file + ".tmp"
        with open(tmp_signal, "w") as f:
            json.dump(metadata, f)
        os.replace(tmp_signal, self.signal_file)
        logger.debug(f"Buffer signaled ready: {self.instance_id}")

    def to_device(self, device: th.device):
        self.device = device
        return self

    def clear_vram_cache(self):
        pass

    def check_ready(self) -> Optional[Dict[str, Any]]:
        """
        Learner calls this to see if the Actor is done.
        """
        if os.path.exists(self.signal_file):
            try:
                with open(self.signal_file, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def clear_signal(self):
        """
        Learner calls this after it has finished consuming the data.
        """
        if os.path.exists(self.signal_file):
            os.remove(self.signal_file)
        logger.debug(f"Buffer signal cleared: {self.instance_id}")

    @staticmethod
    def attach_disk_observations(
        buffer: "AsyncDiskBuffer",
        instance_dir: str,
        meta: Dict[str, Any],
    ) -> None:
        """
        Learner: (re)open collector memmap files for map/tactical.

        Must run on **every** consume. Reusing a prior ``np.memmap`` handle across
        async buffer cycles can serve stale pages (Windows cross-process), desyncing
        observations from ``old_log_prob`` (approx_kl spikes, NaN loss).
        """
        buffer_size = int(meta["buffer_size"])
        n_envs = int(meta["n_envs"])
        if not hasattr(buffer, "disk_files"):
            buffer.disk_files = {}
        # Attached files belong to the collector's static buffer directory.
        buffer._owns_disk_files = False

        disk_meta = meta.get("disk_obs") or {}
        for key, subspace in buffer.observation_space.spaces.items():
            if not is_disk_offloaded_obs_key(key):
                continue
            filename = os.path.join(instance_dir, f"obs_{key}.bin")
            if not os.path.exists(filename):
                logger.warning(f"Disk buffer IPC: missing {filename}")
                continue
            existing = buffer.observations.get(key)
            if isinstance(existing, np.memmap):
                try:
                    existing._mmap.close()
                except Exception:
                    pass
            shape = (buffer_size, n_envs, *subspace.shape)
            mapped = np.memmap(
                filename, dtype=np.float32, mode="r+", shape=shape
            )
            assert_obs_fingerprint_matches(
                key, mapped, disk_meta.get(key) or {}, kind="Disk"
            )
            buffer.observations[key] = mapped
            buffer.disk_files[key] = filename

    def wait_for_consumer(self, timeout=60):
        """
        Actor calls this to wait until the Learner has finished with this buffer 
        before overwriting it. This prevents 'Staleness Overrun'.
        """
        start_time = time.time()
        while os.path.exists(self.signal_file):
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for Learner to consume buffer {self.instance_id}")
                return False
            time.sleep(0.1)
        return True

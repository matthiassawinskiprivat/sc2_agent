import os
import numpy as np
import torch as th
import json
import time
from typing import Dict, Any, Optional, Union, Tuple
from gymnasium import spaces
from loguru import logger
from src.agent.utils.disk_buffer import DiskDictRolloutBuffer

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
        # Save RAM-based payload to disk for the Learner
        payload_path = os.path.join(self.instance_dir, "payload.npz")
        
        # Collect all observations that are NOT in memmaps
        ram_observations = {}
        for key, obs in self.observations.items():
            if not isinstance(obs, np.memmap):
                ram_observations[f"obs_{key}"] = obs

        # Compressed npz: much less disk traffic than raw savez (major win on system drive).
        # Temp name must end with `.npz` — NumPy appends `.npz` if the path does not, so
        # `payload.npz.tmp` becomes `payload.npz.tmp.npz` and os.replace fails on Windows.
        tmp_payload = os.path.join(self.instance_dir, "payload_staging.npz")
        np.savez_compressed(
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
            **ram_observations
        )
        os.replace(tmp_payload, payload_path)

        # Save structural metadata so the Learner knows exactly how to read the binary file
        metadata["buffer_size"] = self.buffer_size
        metadata["n_envs"] = self.n_envs
        metadata["observation_space"] = {k: v.shape for k, v in self.observation_space.spaces.items()}

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

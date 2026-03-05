import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger

class ImitationReplayBuffer:
    """
    Loads and stores expert transitions from .npz files for imitation learning.
    Supports both individual transition sampling and sequence sampling.
    """
    def __init__(self, level_name: str, project_root: str = "."):
        self.level_name = level_name
        self.project_root = project_root
        self.translated_dir = os.path.join(project_root, "imitation_learning", "replays_translated", level_name)
        
        self.replays: List[Dict[str, np.ndarray]] = []
        self.total_size = 0
        
        self.load_all_data()

    def load_all_data(self):
        """Scans the translated directory and loads all .npz files."""
        if not os.path.exists(self.translated_dir):
            logger.info(f"No imitation data found for level {self.level_name}")
            return

        npz_files = glob.glob(os.path.join(self.translated_dir, "*.npz"))
        for f in npz_files:
            try:
                data = np.load(f)
                replay_data = {
                    "observations": data["observations"],
                    "actions": data["actions"],
                    "action_masks": data.get("action_masks")
                }
                self.replays.append(replay_data)
                self.total_size += len(replay_data["actions"])
            except Exception as e:
                logger.error(f"Failed to load imitation file {f}: {e}")

        if self.replays:
            logger.info(f"Loaded {self.total_size} expert transitions from {len(self.replays)} replays for {self.level_name}")

    def sample_batch(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """Samples a random batch of (Obs, Action, Mask) pairs from all replays."""
        if self.total_size == 0:
            return None
        
        # Flattened access for legacy/random sampling
        all_obs = np.concatenate([r["observations"] for r in self.replays], axis=0)
        all_acts = np.concatenate([r["actions"] for r in self.replays], axis=0)
        
        indices = np.random.choice(len(all_acts), min(batch_size, len(all_acts)), replace=False)
        
        masks = None
        if self.replays[0]["action_masks"] is not None:
            all_masks = np.concatenate([r["action_masks"] for r in self.replays], axis=0)
            masks = all_masks[indices]
            
        return all_obs[indices], all_acts[indices], masks

    def sample_sequences(self, num_sequences: int, sequence_length: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Samples a batch of sequences for recurrent training.
        Returns:
            Dict containing 'observations', 'actions', 'action_masks', 'episode_starts'
            Each has shape (num_sequences, sequence_length, ...)
        """
        if not self.replays:
            return None

        sampled_obs = []
        sampled_acts = []
        sampled_masks = []
        sampled_starts = []

        for _ in range(num_sequences):
            # 1. Pick a random replay
            replay = self.replays[np.random.randint(len(self.replays))]
            n = len(replay["actions"])
            
            # 2. Pick a start index
            # If replay is shorter than sequence_length, we take the whole thing and pad
            start_idx = np.random.randint(0, max(1, n - sequence_length + 1))
            end_idx = min(start_idx + sequence_length, n)
            
            obs = replay["observations"][start_idx:end_idx]
            acts = replay["actions"][start_idx:end_idx]
            masks = replay["action_masks"][start_idx:end_idx] if replay["action_masks"] is not None else None
            
            # Episode start is True only for the very first step of a replay
            starts = np.zeros(len(obs), dtype=bool)
            if start_idx == 0:
                starts[0] = True
            
            # 3. Padding if necessary
            if len(obs) < sequence_length:
                pad_len = sequence_length - len(obs)
                obs = np.concatenate([obs, np.tile(obs[-1:], (pad_len, 1, 1, 1))], axis=0)
                
                # Robust padding for actions (handles both scalars and MultiDiscrete vectors)
                last_act = acts[-1:]
                act_pad_shape = [pad_len] + [1] * (len(last_act.shape) - 1)
                acts = np.concatenate([acts, np.tile(last_act, tuple(act_pad_shape))], axis=0)
                
                starts = np.concatenate([starts, np.zeros(pad_len, dtype=bool)], axis=0)
                if masks is not None:
                    masks = np.concatenate([masks, np.tile(masks[-1:], (pad_len, 1))], axis=0)

            sampled_obs.append(obs)
            sampled_acts.append(acts)
            sampled_starts.append(starts)
            if masks is not None:
                sampled_masks.append(masks)

        return {
            "observations": np.array(sampled_obs),
            "actions": np.array(sampled_acts),
            "episode_starts": np.array(sampled_starts),
            "action_masks": np.array(sampled_masks) if sampled_masks else None
        }

    def sample_full_episode(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Samples a full episode for state-preserving recurrent training.
        Returns:
            Dict containing 'observations', 'actions', 'action_masks', 'episode_starts'
            Each has shape (1, sequence_length, ...) - Batch size of 1
        """
        if not self.replays:
            return None

        # 1. Pick a random replay
        replay = self.replays[np.random.randint(len(self.replays))]
        
        # 2. Extract full arrays
        obs = replay["observations"]
        acts = replay["actions"]
        masks = replay["action_masks"]
        
        n = len(acts)

        # 3. Create Episode Starts (True at index 0, False elsewhere)
        starts = np.zeros(n, dtype=bool)
        starts[0] = True

        # 4. Return with Batch Dimension (1)
        # PPO expects (Batch, Seq, ...)
        return {
            "observations": obs[np.newaxis, ...],   # (1, T, ...)
            "actions": acts[np.newaxis, ...],       # (1, T)
            "episode_starts": starts[np.newaxis, ...], # (1, T)
            "action_masks": masks[np.newaxis, ...] if masks is not None else None # (1, T, ...)
        }

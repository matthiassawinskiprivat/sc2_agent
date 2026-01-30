import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger

class ImitationReplayBuffer:
    """
    Loads and stores expert transitions from .npz files for imitation learning.
    """
    def __init__(self, level_name: str, project_root: str = "."):
        self.level_name = level_name
        self.project_root = project_root
        self.translated_dir = os.path.join(project_root, "imitation_learning", "replays_translated", level_name)
        
        self.observations = None
        self.actions = None
        self.total_size = 0
        
        self.load_all_data()

    def load_all_data(self):
        """Scans the translated directory and loads all .npz files."""
        if not os.path.exists(self.translated_dir):
            logger.info(f"No imitation data found for level {self.level_name}")
            return

        all_obs = []
        all_acts = []
        
        npz_files = glob.glob(os.path.join(self.translated_dir, "*.npz"))
        for f in npz_files:
            try:
                data = np.load(f)
                all_obs.append(data["observations"])
                all_acts.append(data["actions"])
            except Exception as e:
                logger.error(f"Failed to load imitation file {f}: {e}")

        if all_obs:
            self.observations = np.concatenate(all_obs, axis=0)
            self.actions = np.concatenate(all_acts, axis=0)
            self.total_size = len(self.actions)
            logger.info(f"Loaded {self.total_size} expert transitions for {self.level_name}")

    def sample_batch(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Samples a random batch of (Obs, Action) pairs."""
        if self.total_size == 0:
            return None
        
        indices = np.random.choice(self.total_size, min(batch_size, self.total_size), replace=False)
        return self.observations[indices], self.actions[indices]

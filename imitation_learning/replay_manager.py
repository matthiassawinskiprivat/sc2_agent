import os
import datetime
from pathlib import Path
from typing import List, Dict

class ReplayManager:
    """
    Manages raw and translated replays on disk.
    """
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.raw_dir = self.project_root / "imitation_learning" / "replays_raw"
        self.translated_dir = self.project_root / "imitation_learning" / "replays_translated"

    def list_raw_replays(self, level_name: str) -> List[Dict]:
        """Lists raw replays for a level with metadata."""
        level_dir = self.raw_dir / level_name
        if not level_dir.exists():
            return []
        
        replays = []
        for f in level_dir.glob("*.SC2Replay"):
            stats = f.stat()
            replays.append({
                "name": f.name,
                "path": str(f),
                "date": datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": f"{stats.st_size / 1024 / 1024:.2f} MB"
            })
        
        # Sort by date descending
        return sorted(replays, key=lambda x: x["date"], reverse=True)

    def delete_raw_replay(self, level_name: str, filename: str):
        """Deletes a raw replay file."""
        file_path = self.raw_dir / level_name / filename
        if file_path.exists():
            file_path.unlink()

    def list_translated_replays(self, level_name: str) -> List[Dict]:
        """Lists translated .npz datasets for a level."""
        level_dir = self.translated_dir / level_name
        if not level_dir.exists():
            return []
        
        replays = []
        for f in level_dir.glob("*.npz"):
            stats = f.stat()
            replays.append({
                "name": f.name,
                "path": str(f),
                "date": datetime.datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": f"{stats.st_size / 1024 / 1024:.2f} MB"
            })
        return sorted(replays, key=lambda x: x["date"], reverse=True)

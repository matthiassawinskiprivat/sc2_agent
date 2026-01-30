import pytest
import os
from pathlib import Path
from imitation_learning.replay_manager import ReplayManager

def test_replay_manager_listing(tmp_path):
    # Setup mock directories
    raw_dir = tmp_path / "imitation_learning" / "replays_raw" / "level_0"
    raw_dir.mkdir(parents=True)
    (raw_dir / "replay1.SC2Replay").touch()
    
    trans_dir = tmp_path / "imitation_learning" / "replays_translated" / "level_0"
    trans_dir.mkdir(parents=True)
    (trans_dir / "data1.npz").touch()
    
    manager = ReplayManager(project_root=str(tmp_path))
    
    raw_list = manager.list_raw_replays("level_0")
    assert len(raw_list) == 1
    assert raw_list[0]["name"] == "replay1.SC2Replay"
    
    trans_list = manager.list_translated_replays("level_0")
    assert len(trans_list) == 1
    assert trans_list[0]["name"] == "data1.npz"

def test_replay_manager_delete(tmp_path):
    raw_dir = tmp_path / "imitation_learning" / "replays_raw" / "level_0"
    raw_dir.mkdir(parents=True)
    replay_file = raw_dir / "to_delete.SC2Replay"
    replay_file.touch()
    
    manager = ReplayManager(project_root=str(tmp_path))
    assert replay_file.exists()
    
    manager.delete_raw_replay("level_0", "to_delete.SC2Replay")
    assert not replay_file.exists()

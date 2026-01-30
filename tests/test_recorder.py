import pytest
from pathlib import Path
from sc2.data import Race
from imitation_learning.recorder import Recorder

def test_race_parsing():
    recorder = Recorder()
    
    # Test typical map names
    r1, r2 = recorder._parse_races_from_map_name("00_64x64_Terran_vs_Zerg")
    assert r1 == Race.Terran
    assert r2 == Race.Zerg
    
    r1, r2 = recorder._parse_races_from_map_name("01_Protoss_vs_Terran")
    assert r1 == Race.Protoss
    assert r2 == Race.Terran
    
    # Test fallback
    r1, r2 = recorder._parse_races_from_map_name("NoRaceInfo")
    assert r1 == Race.Terran
    assert r2 == Race.Terran

def test_map_path_resolution(tmp_path):
    recorder = Recorder(project_root=str(tmp_path))
    
    # Mock map file
    map_dir = tmp_path / "train" / "maps" / "level_drop_2"
    map_dir.mkdir(parents=True)
    map_file = map_dir / "test_map.SC2Map"
    map_file.touch()
    
    path = recorder.get_map_path("level_drop_2", "test_map")
    assert path.exists()
    assert path.name == "test_map.SC2Map"

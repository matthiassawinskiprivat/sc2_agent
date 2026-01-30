import pytest
import os
import shutil
import yaml
from imitation_learning.config_resolver import ConfigResolver

@pytest.fixture
def mock_project_structure(tmp_path):
    # Create mock config dir
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    pipeline_conf = {'levels': ['level_0', 'level_1']}
    (config_dir / "pipeline.yaml").write_text(yaml.dump(pipeline_conf))
    
    l0_conf = {'env': {'active_layers': ['layer_0'], 'action_heads': ['move']}}
    (config_dir / "level_0.yaml").write_text(yaml.dump(l0_conf))
    
    l1_conf = {'env': {'active_layers': ['layer_1'], 'action_heads': ['attack']}}
    (config_dir / "level_1.yaml").write_text(yaml.dump(l1_conf))
    
    # Create mock maps dir
    maps_dir = tmp_path / "train" / "maps"
    os.makedirs(maps_dir)
    
    l0_maps_dir = maps_dir / "level_0"
    l0_maps_dir.mkdir()
    (l0_maps_dir / "map_a.SC2Map").touch()
    
    l1_maps_dir = maps_dir / "level_1"
    l1_maps_dir.mkdir()
    (l1_maps_dir / "map_b.SC2Map").touch()
    (l1_maps_dir / "map_c.SC2Map").touch()
    
    return tmp_path

def test_list_levels(mock_project_structure):
    resolver = ConfigResolver(project_root=str(mock_project_structure))
    levels = resolver.list_levels()
    # Should find level_0 and level_1 (from pipeline or directory)
    assert "level_0" in levels
    assert "level_1" in levels
    assert "pipeline" not in levels

def test_list_maps(mock_project_structure):
    resolver = ConfigResolver(project_root=str(mock_project_structure))
    
    l0_maps = resolver.list_maps("level_0")
    assert len(l0_maps) == 1
    assert "map_a" in l0_maps
    
    l1_maps = resolver.list_maps("level_1")
    assert len(l1_maps) == 2
    assert "map_b" in l1_maps
    assert "map_c" in l1_maps

def test_get_cumulative_config(mock_project_structure):
    resolver = ConfigResolver(project_root=str(mock_project_structure))
    
    config = resolver.get_cumulative_config("level_1")
    assert "active_layers" in config
    assert "action_heads" in config
    
    assert "layer_0" in config["active_layers"]
    assert "layer_1" in config["active_layers"]
    assert "move" in config["action_heads"]
    assert "attack" in config["action_heads"]

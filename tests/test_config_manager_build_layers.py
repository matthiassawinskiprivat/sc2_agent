from src.utils.config_manager import ConfigManager


def test_cumulative_build_map_layers(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        "build_map_layers: [terrain_pathable]\nlevels: [lvl0, lvl1]\n"
    )
    (tmp_path / "lvl0.yaml").write_text("env:\n  build_map_layers: [terrain_creep]\n")
    (tmp_path / "lvl1.yaml").write_text(
        "env:\n  build_map_layers: [terrain_buildable]\n"
    )

    cm = ConfigManager(str(tmp_path))
    assert cm.get_cumulative_build_map_layers("lvl1") == [
        "terrain_pathable",
        "terrain_creep",
        "terrain_buildable",
    ]


def test_cumulative_build_commands(tmp_path):
    (tmp_path / "pipeline.yaml").write_text(
        "build_commands: [extractor]\nlevels: [lvl0, lvl1]\n"
    )
    (tmp_path / "lvl0.yaml").write_text("env:\n  build_commands: [spawningpool]\n")
    (tmp_path / "lvl1.yaml").write_text("env:\n  build_commands: [zergling]\n")

    cm = ConfigManager(str(tmp_path))
    assert cm.get_cumulative_build_commands("lvl1") == [
        "extractor",
        "spawningpool",
        "zergling",
    ]

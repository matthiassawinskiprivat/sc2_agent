"""``use_relational_decode`` is a per-level YAML flag, like ``use_self_attention``."""

from src.utils.config_manager import ConfigManager


RELATIONAL_DECODE_START_LEVEL = "ctf_level_17"


def _flag(level_conf) -> bool:
    fe_kw = (
        (level_conf.get("model") or {})
        .get("policy_kwargs", {})
        .get("features_extractor_kwargs", {})
        or {}
    )
    return bool(fe_kw.get("use_relational_decode", False))


def test_yaml_flag_starts_at_ctf_level_17():
    cm = ConfigManager("config_universal")
    start = cm.levels.index(RELATIONAL_DECODE_START_LEVEL)
    for name in cm.levels[:start]:
        assert _flag(cm.get_level_config(name)) is False, name
    for name in cm.levels[start:]:
        assert _flag(cm.get_level_config(name)) is True, name


def test_neighbors_of_cutoff():
    cm = ConfigManager("config_universal")
    assert _flag(cm.get_level_config("ctf_level_16")) is False
    assert _flag(cm.get_level_config("ctf_level_16_reversed")) is False
    assert _flag(cm.get_level_config("ctf_level_17")) is True
    assert _flag(cm.get_level_config("ctf_level_17_reversed")) is True
    assert _flag(cm.get_level_config("ctf_level_17_mirror_terran")) is True

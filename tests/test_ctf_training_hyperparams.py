"""CaptureTheFlagEnv curriculum shares one training throughput template."""
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_universal"

_EXPECTED = {
    "num_envs": 6,
    "n_steps": 1024,
    "batch_size": 256,
    "rollout_buffer": "disk",
}


def _ctf_yaml_paths():
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        env = data.get("env") or {}
        if env.get("env_class") == "CaptureTheFlagEnv":
            yield path, data


def test_all_capture_the_flag_levels_share_training_template():
    found = list(_ctf_yaml_paths())
    assert found, "expected CaptureTheFlagEnv YAML under config_universal"
    for path, data in found:
        training = data.get("training") or {}
        for key, expected in _EXPECTED.items():
            assert training.get(key) == expected, f"{path.name} {key}"

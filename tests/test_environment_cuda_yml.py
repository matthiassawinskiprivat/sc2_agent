"""Sanity checks for GPU conda env file (no conda required)."""

from pathlib import Path

import yaml


def test_environment_cuda_yml_exists_and_names_env():
    path = Path(__file__).resolve().parents[1] / "environment_cuda.yml"
    assert path.is_file(), "environment_cuda.yml should exist at repo root"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data.get("name") == "sc2_agent_cuda"
    deps = data.get("dependencies") or []
    dep_strs = [d for d in deps if isinstance(d, str)]
    assert any("pytorch-cuda=12.1" in d for d in dep_strs), deps
    pip_block = next((d for d in deps if isinstance(d, dict) and "pip" in d), None)
    assert pip_block is not None
    pip_lines = "\n".join(pip_block["pip"])
    assert "torch==1.13.1" not in pip_lines
    assert "stable-baselines3" in pip_lines

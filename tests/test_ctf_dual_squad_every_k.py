"""Regression: dual-squad CTF uses the unified every_k_episodes schedule."""
from pathlib import Path

import yaml

from train.envs.sc2_session_config import every_k_episodes


def test_dual_squad_ctf_level_03_uses_unified_every_k():
    """All CTF squad levels share every_k_episodes: 200 after the hparam unify."""
    cfg = yaml.safe_load(
        (Path("config_universal") / "ctf_level_03.yaml").read_text(encoding="utf-8")
    )
    assert every_k_episodes(cfg) == 200

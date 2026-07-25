"""Regression: dual-squad CTF hard-resets SC2 often enough to bound process GPU growth."""
from train.envs.sc2_session_config import every_k_episodes


def test_dual_squad_ctf_level_03_hard_resets_frequently():
    """
    ctf_level_03 runs two trainee pools (4 armies). Soft-reset sessions that last
    100 episodes can fill dedicated GPU memory while torch.cuda.* stays flat.
    Keep every_k low so SC2 is relaunched before that window.
    """
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path("config_universal") / "ctf_level_03.yaml").read_text(encoding="utf-8")
    )
    k = every_k_episodes(cfg)
    assert k <= 25, f"ctf_level_03 every_k_episodes={k} too high for dual-squad"

"""Rollout policy snapshot must load policy weights (not fail via set_parameters exact_match)."""
import tempfile

import torch

from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.models.squad_network import SquadNetwork
from src.agent.models.squad_policy import SquadRecurrentPolicy
from src.utils.config_manager import ConfigManager
from src.utils.model_utils import load_policy_weights_from_zip
from train.env_utils import create_dummy_env_from_config


def test_load_policy_weights_from_zip_roundtrip():
    cm = ConfigManager("config_universal")
    cfg = {
        "config_dir": "config_universal",
        "level_name": "ctf_level_01",
        "active_layers": cm.get_cumulative_active_layers("ctf_level_01"),
        "active_tactical_layers": cm.get_cumulative_active_tactical_layers("ctf_level_01"),
        "active_unit_scalars": cm.get_cumulative_active_unit_scalars("ctf_level_01"),
        "action_heads": cm.get_cumulative_action_heads("ctf_level_01"),
        "env": {"include_race": False},
        "model": {
            "algorithm": "MaskableRecurrentPPO",
            "rnn_type": "gru",
            "policy_kwargs": {
                "features_extractor_class": SquadNetwork,
                "features_extractor_kwargs": {"features_dim": 64, "thinking_steps": 1},
                "lstm_hidden_size": 32,
                "spatial_resolution": 32,
            },
        },
    }
    env = create_dummy_env_from_config(cfg, level_name="ctf_level_01")
    m1 = MaskableRecurrentPPO(
        SquadRecurrentPolicy,
        env,
        n_steps=64,
        batch_size=32,
        policy_kwargs=cfg["model"]["policy_kwargs"],
        device="cpu",
    )
    assert hasattr(m1.policy.features_extractor, "enemy_encoder")
    assert m1.policy.features_extractor.enemy_encoder is not None
    w_before = m1.policy.command_head.weight.detach().clone()
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/snap.zip"
        m1.save(path)
        m2 = MaskableRecurrentPPO(
            SquadRecurrentPolicy,
            env,
            n_steps=64,
            batch_size=32,
            policy_kwargs=cfg["model"]["policy_kwargs"],
            device="cpu",
        )
        ok, missing, _ = load_policy_weights_from_zip(m2.policy, path, "cpu")
        assert ok and not missing
        assert torch.allclose(w_before, m2.policy.command_head.weight.detach())

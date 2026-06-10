"""Rollout policy snapshot must load policy weights (not fail via set_parameters exact_match)."""
import tempfile

import torch

from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.models.universal_recurrent_policy import UniversalRecurrentPolicy
from src.agent.models.unit_control_network import UnitControlNetwork
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
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
        "action_heads": cm.get_cumulative_action_heads("ctf_level_01"),
        "model": {
            "algorithm": "MaskableRecurrentPPO",
            "rnn_type": "gru",
            "policy_kwargs": {
                "features_extractor_class": UnitControlNetwork,
                "features_extractor_kwargs": {"features_dim": 64, "thinking_steps": 1},
                "lstm_hidden_size": 32,
            },
        },
    }
    env = create_dummy_env_from_config(cfg, level_name="ctf_level_01")
    m1 = MaskableRecurrentPPO(
        UniversalRecurrentPolicy,
        env,
        n_steps=64,
        batch_size=32,
        policy_kwargs=cfg["model"]["policy_kwargs"],
        device="cpu",
    )
    w_before = m1.policy.command_head.weight.detach().clone()
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/snap.zip"
        m1.save(path)
        m2 = MaskableRecurrentPPO(
            UniversalRecurrentPolicy,
            env,
            n_steps=64,
            batch_size=32,
            policy_kwargs=cfg["model"]["policy_kwargs"],
            device="cpu",
        )
        ok, missing, _ = load_policy_weights_from_zip(m2.policy, path, "cpu")
        assert ok and not missing
        assert torch.allclose(w_before, m2.policy.command_head.weight.detach())

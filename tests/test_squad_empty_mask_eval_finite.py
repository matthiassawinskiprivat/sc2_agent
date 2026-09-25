"""Empty enemy/ally masks must stay finite in eval (async rollout collection)."""
from pathlib import Path

import pytest
import torch
import yaml
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.models.squad_network import SquadNetwork
from src.agent.models.squad_policy import SquadRecurrentPolicy
from src.utils.model_utils import load_policy_weights_from_zip
from train.env_utils import create_dummy_env_from_config

_THOR_CKPT = Path(
    "models_universal/ctf_heal_thor_scv_vs_thor_scv/universal_153600_steps.zip"
)


@pytest.mark.skipif(not _THOR_CKPT.is_file(), reason="thor heal checkpoint not present")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_thor_checkpoint_empty_enemy_eval_finite():
    level = "ctf_heal_thor_scv_vs_thor_scv"
    with open(f"config_universal/{level}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["agent_architecture"] = "squad"
    cfg["config_dir"] = "config_universal"
    venv = DummyVecEnv([lambda: create_dummy_env_from_config(cfg, level_name=level)])
    model = MaskableRecurrentPPO(
        SquadRecurrentPolicy,
        venv,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": SquadNetwork,
            "features_extractor_kwargs": {
                "features_dim": 512,
                "thinking_steps": 3,
                "use_self_attention": True,
            },
            "lstm_hidden_size": 512,
            "net_arch": [],
        },
        device="cuda",
        verbose=0,
        rnn_type="gru",
    )
    ok, _, _ = load_policy_weights_from_zip(model.policy, str(_THOR_CKPT), strict=False)
    assert ok
    policy = model.policy.eval()

    obs = venv.reset()
    batch = 8
    o = {
        k: torch.as_tensor(v, device="cuda")
        for k, v in obs.items()
    }
    o = {
        k: (
            v.float() if v.is_floating_point() or v.dtype == torch.float64 else v
        ).repeat(batch, *([1] * (v.dim() - 1)))
        for k, v in o.items()
    }
    o["enemy_mask"] = torch.zeros_like(o["enemy_mask"])
    o["unit_mask"] = torch.zeros_like(o["unit_mask"])
    o["unit_mask"][:, :2] = 1.0
    am = (
        torch.as_tensor(venv.env_method("action_masks")[0], device="cuda")
        .bool()
        .unsqueeze(0)
        .repeat(batch, 1)
    )
    h = torch.zeros(1, batch, 512, device="cuda")
    states = RNNStates((h,), (h.clone(),))
    starts = torch.ones(batch, device="cuda")

    with torch.no_grad():
        _acts, values, logp, new_states = policy.forward(
            o, states, starts, action_masks=am, deterministic=True
        )
    assert torch.isfinite(values).all()
    assert torch.isfinite(logp).all()
    assert all(torch.isfinite(t).all() for t in new_states.pi)
    assert all(torch.isfinite(t).all() for t in new_states.vf)

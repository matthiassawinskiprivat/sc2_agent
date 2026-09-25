"""Rescore live disk buffer against its rollout_policy.zip."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_schedule_fn

from src.agent.models.squad_policy import SquadRecurrentPolicy
from src.agent.utils.disk_buffer import obs_sample_mean
from src.utils.model_utils import load_policy_weights_from_zip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default=r"C:\temp\sc2_async_buffer\buffer_1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    buf = Path(args.buffer)
    device = torch.device(args.device)

    data, _, _ = load_from_zip_file(str(buf / "rollout_policy.zip"), device="cpu", load_data=True)
    obs_space = data["observation_space"]
    act_space = data["action_space"]
    pk = dict(data["policy_kwargs"])
    pk.pop("features_extractor_class", None)
    pk["features_extractor_class"] = data["policy_kwargs"]["features_extractor_class"]

    policy = SquadRecurrentPolicy(
        obs_space,
        act_space,
        get_schedule_fn(float(data.get("learning_rate", 3e-4))),
        **pk,
    )
    ok, missing, unexpected = load_policy_weights_from_zip(
        policy, str(buf / "rollout_policy.zip"), device, strict=False
    )
    policy.to(device)
    policy.set_training_mode(False)
    print(f"load ok={ok} missing={missing[:8]} unexpected={unexpected[:8]}")

    z = np.load(buf / "payload.npz", allow_pickle=True)
    n_steps, n_envs = z["actions"].shape[:2]
    map_shape = obs_space["map"].shape
    tac_shape = obs_space["anchor_tactical"].shape
    obs_map = np.memmap(
        buf / "obs_map.bin", dtype=np.float32, mode="r", shape=(n_steps, n_envs, *map_shape)
    )
    obs_tac = np.memmap(
        buf / "obs_anchor_tactical.bin",
        dtype=np.float32,
        mode="r",
        shape=(n_steps, n_envs, *tac_shape),
    )
    print(
        "fingerprint map",
        obs_sample_mean(obs_map),
        "tac",
        obs_sample_mean(obs_tac),
        "unit",
        float(z["obs_unit_scalars"][0, 0].mean()),
    )

    rng = np.random.default_rng(0)
    pairs = [(t, e) for t in range(n_steps) for e in range(n_envs)]
    if len(pairs) > args.n:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=args.n, replace=False)]

    errs = []
    parts = {"cmd": [], "actors": [], "target": [], "spatial": [], "value": []}
    with torch.no_grad():
        for t, e in pairs:
            obs = {
                "map": torch.as_tensor(np.asarray(obs_map[t, e])[None].copy(), device=device),
                "anchor_tactical": torch.as_tensor(
                    np.asarray(obs_tac[t, e])[None].copy(), device=device
                ),
                "command": torch.as_tensor(z["obs_command"][t, e][None].copy(), device=device),
                "command_modes": torch.as_tensor(
                    z["obs_command_modes"][t, e][None].copy(), device=device
                ),
                "unit_scalars": torch.as_tensor(
                    z["obs_unit_scalars"][t, e][None].copy(), device=device
                ),
                "unit_mask": torch.as_tensor(z["obs_unit_mask"][t, e][None].copy(), device=device),
                "enemy_scalars": torch.as_tensor(
                    z["obs_enemy_scalars"][t, e][None].copy(), device=device
                ),
                "enemy_mask": torch.as_tensor(
                    z["obs_enemy_mask"][t, e][None].copy(), device=device
                ),
                "heal_actor_mask": torch.as_tensor(
                    z["obs_heal_actor_mask"][t, e][None].copy(), device=device
                ),
                "heal_target_mask": torch.as_tensor(
                    z["obs_heal_target_mask"][t, e][None].copy(), device=device
                ),
                "load_actor_mask": torch.as_tensor(
                    z["obs_load_actor_mask"][t, e][None].copy(), device=device
                ),
                "load_target_mask": torch.as_tensor(
                    z["obs_load_target_mask"][t, e][None].copy(), device=device
                ),
                "attack_actor_mask": torch.as_tensor(
                    z["obs_attack_actor_mask"][t, e][None].copy(), device=device
                ),
            }
            acts = torch.as_tensor(z["actions"][t, e][None].copy(), device=device).long()
            starts = torch.as_tensor(
                [float(z["episode_starts"][t, e])], device=device, dtype=torch.float32
            )
            masks = torch.as_tensor(z["action_masks"][t, e][None].copy(), device=device)
            h_pi = torch.as_tensor(
                z["hidden_states_pi"][t, :, e, :][:, None, :].copy(), device=device
            )
            h_vf = torch.as_tensor(
                z["hidden_states_vf"][t, :, e, :][:, None, :].copy(), device=device
            )
            states = RNNStates((h_pi,), (h_vf,))
            vals, new_lp, _ = policy.evaluate_actions(obs, acts, states, starts, masks)
            old = float(z["log_probs"][t, e])
            err = abs(float(new_lp.reshape(-1)[0]) - old)
            errs.append(err)
            old_v = float(z["values"][t, e])
            parts["value"].append(abs(float(vals.reshape(-1)[0]) - old_v))
            fe = getattr(policy, "features_extractor", None)
            if fe is not None and hasattr(fe, "clear_forward_cache"):
                fe.clear_forward_cache()

    errs_a = np.asarray(errs)
    print(
        f"abs_err mean={errs_a.mean():.6f} median={np.median(errs_a):.6f} "
        f"p90={np.percentile(errs_a,90):.6f} max={errs_a.max():.6f} "
        f"frac>0.01={(errs_a>0.01).mean():.3f} frac>0.1={(errs_a>0.1).mean():.3f}"
    )
    print(
        f"value abs_err mean={np.mean(parts['value']):.6f} "
        f"max={np.max(parts['value']):.6f}"
    )


if __name__ == "__main__":
    main()

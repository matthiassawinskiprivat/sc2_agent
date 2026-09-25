"""Reproduce squad collect vs evaluate log-prob through disk buffer + RNN sequences."""
from __future__ import annotations

import tempfile

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import get_schedule_fn

from src.agent.models.squad_policy import SquadRecurrentPolicy
from src.agent.squad_constants import (
    CMD_MODE_ALLY_TARGET,
    CMD_MODE_ENEMY_TARGET,
    CMD_MODE_LOAD_TARGET,
    CMD_MODE_NONE,
    CMD_MODE_SPATIAL,
    MAX_SQUAD_ENEMIES,
    MAX_SQUAD_UNITS,
    SQUAD_FLAT_ACTION_DIM,
    SQUAD_MASK_COMMAND_OFFSET,
    squad_action_space_nvec,
)
from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM
from src.agent.utils.disk_buffer import DiskDictRolloutBuffer

N_FEAT = 6
N_LAYERS = 3
HIDDEN = 32
COMMAND_MODES = [
    CMD_MODE_SPATIAL,
    CMD_MODE_ENEMY_TARGET,
    CMD_MODE_ALLY_TARGET,
    CMD_MODE_LOAD_TARGET,
]


def obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(0, 1, (N_LAYERS, 128, 128), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
            "anchor_tactical": spaces.Box(0, 1, (N_LAYERS, 32, 32), dtype=np.float32),
            "unit_scalars": spaces.Box(0, 1, (MAX_SQUAD_UNITS, N_FEAT), dtype=np.float32),
            "unit_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "enemy_scalars": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES, N_FEAT), dtype=np.float32),
            "enemy_mask": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES,), dtype=np.float32),
            "command_modes": spaces.Box(0, 4, (FIXED_UNIVERSAL_COMMAND_DIM,), dtype=np.float32),
            "heal_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "heal_target_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "load_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "load_target_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "attack_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
        }
    )


def make_obs(n_alive=3, n_enemies=2, seed=0):
    rng = np.random.default_rng(seed)
    unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unit_mask[:n_alive] = 1.0
    enemy_mask = np.zeros(MAX_SQUAD_ENEMIES, dtype=np.float32)
    enemy_mask[:n_enemies] = 1.0
    heal_actor = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    heal_actor[0] = 1.0
    heal_target = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    heal_target[1] = 1.0
    modes = np.full(FIXED_UNIVERSAL_COMMAND_DIM, float(CMD_MODE_NONE), dtype=np.float32)
    for i, m in enumerate(COMMAND_MODES):
        modes[i] = float(m)
    return {
        "map": rng.random((N_LAYERS, 128, 128), dtype=np.float32),
        "command": rng.random(10, dtype=np.float32),
        "anchor_tactical": rng.random((N_LAYERS, 32, 32), dtype=np.float32),
        "unit_scalars": rng.random((MAX_SQUAD_UNITS, N_FEAT), dtype=np.float32)
        * unit_mask[:, None],
        "unit_mask": unit_mask,
        "enemy_scalars": rng.random((MAX_SQUAD_ENEMIES, N_FEAT), dtype=np.float32)
        * enemy_mask[:, None],
        "enemy_mask": enemy_mask,
        "command_modes": modes,
        "heal_actor_mask": heal_actor,
        "heal_target_mask": heal_target,
        "load_actor_mask": np.zeros(MAX_SQUAD_UNITS, dtype=np.float32),
        "load_target_mask": np.zeros(MAX_SQUAD_UNITS, dtype=np.float32),
        "attack_actor_mask": unit_mask.copy(),
    }


def flat_mask(n_alive=3):
    mask = np.ones(SQUAD_FLAT_ACTION_DIM, dtype=bool)
    for slot in range(n_alive, MAX_SQUAD_UNITS):
        mask[slot * 2] = True
        mask[slot * 2 + 1] = False
    cmd = SQUAD_MASK_COMMAND_OFFSET
    mask[cmd : cmd + FIXED_UNIVERSAL_COMMAND_DIM] = False
    mask[cmd : cmd + len(COMMAND_MODES)] = True
    return mask


def main(tmp: str):
    policy = SquadRecurrentPolicy(
        obs_space(),
        spaces.MultiDiscrete(squad_action_space_nvec()),
        get_schedule_fn(3e-4),
        rnn_type="gru",
        lstm_hidden_size=HIDDEN,
        features_extractor_kwargs={"features_dim": HIDDEN, "thinking_steps": 1},
        net_arch=[],
    )
    policy.set_training_mode(False)

    n_envs, n_steps = 2, 64
    buf = DiskDictRolloutBuffer(
        n_steps,
        obs_space(),
        spaces.MultiDiscrete(squad_action_space_nvec()),
        device="cpu",
        n_envs=n_envs,
        temp_dir=tmp,
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=HIDDEN,
    )

    mask = flat_mask()
    h = torch.zeros(1, n_envs, HIDDEN)
    lstm = RNNStates((h,), (h,))
    for t in range(n_steps):
        row_obs = {k: [] for k in obs_space().spaces}
        for env_i in range(n_envs):
            o = make_obs(seed=t * 10 + env_i)
            for k, v in o.items():
                row_obs[k].append(v)
        batch_obs = {k: np.stack(v) for k, v in row_obs.items()}
        batch_masks = np.stack([mask.copy() for _ in range(n_envs)])
        starts = np.array([1.0 if t == 0 else 0.0] * n_envs, dtype=np.float32)
        pre = RNNStates(
            tuple(s.detach().clone() for s in lstm.pi),
            tuple(s.detach().clone() for s in lstm.vf),
        )
        obs_t = policy.obs_to_tensor(batch_obs)[0]
        with torch.no_grad():
            acts, vals, logp, lstm = policy.forward(
                obs_t,
                pre,
                torch.as_tensor(starts),
                action_masks=torch.as_tensor(batch_masks),
            )
        buf.add(
            batch_obs,
            acts.cpu().numpy(),
            np.zeros(n_envs, dtype=np.float32),
            starts,
            vals.detach(),
            logp.detach(),
            lstm_states=pre,
            action_masks=batch_masks,
            policy_active=np.ones(n_envs, dtype=np.float32),
        )

    buf.full = True
    buf.compute_returns_and_advantage(
        torch.zeros((n_envs, 1)), np.zeros(n_envs, dtype=bool)
    )
    policy.set_training_mode(True)

    errs = []
    for t in range(n_steps):
        obs_t = {k: torch.as_tensor(buf.observations[k][t]) for k in buf.observations}
        acts = torch.as_tensor(buf.actions[t]).long()
        old = torch.as_tensor(buf.log_probs[t])
        starts = torch.as_tensor(buf.episode_starts[t])
        masks = torch.as_tensor(buf.action_masks[t])
        states = RNNStates(
            (torch.as_tensor(buf.hidden_states_pi[t]),),
            (torch.as_tensor(buf.hidden_states_vf[t]),),
        )
        with torch.no_grad():
            _, new_lp, _ = policy.evaluate_actions(obs_t, acts, states, starts, masks)
        errs.append((new_lp - old).abs().cpu().numpy())
    print(f"single-step mean abs logp err = {float(np.mean(np.stack(errs))):.6f}")

    seq_errs = []
    empty = nonfinite_loss = nonfinite_grad = n_batches = 0
    for batch in buf.get(batch_size=64):
        n_batches += 1
        mask_t = batch.mask.reshape(-1) > 1e-8
        if not mask_t.any():
            empty += 1
            continue
        _, log_p, ent = policy.evaluate_actions(
            batch.observations,
            batch.actions.long(),
            batch.lstm_states,
            batch.episode_starts,
            batch.action_masks,
        )
        log_p = log_p.reshape(-1)
        old = batch.old_log_prob.reshape(-1)
        seq_errs.append((log_p - old)[mask_t].abs().mean().item())
        ratio = torch.exp((log_p - old).clamp(-5, 5))
        adv = batch.advantages.reshape(-1)
        adv_sel = adv[mask_t]
        adv = (adv - adv_sel.mean()) / (adv_sel.std() + 1e-8)
        loss = -torch.mean(torch.min(adv * ratio, adv * ratio.clamp(0.8, 1.2))[mask_t])
        vpred = batch.old_values.reshape(-1)
        loss = loss + 0.5 * torch.mean(
            ((batch.returns.reshape(-1) - vpred)[mask_t]) ** 2
        )
        loss = loss + 0.01 * (-torch.mean(ent.reshape(-1)[mask_t]))
        if not torch.isfinite(loss):
            nonfinite_loss += 1
            continue
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        if not torch.isfinite(gn):
            nonfinite_grad += 1
            print(f"  nonfinite grad_norm={gn}")
        policy.optimizer.zero_grad(set_to_none=True)

    print(
        f"sequence batches={n_batches} empty={empty} "
        f"nonfinite_loss={nonfinite_loss} nonfinite_grad={nonfinite_grad}"
    )
    print(f"sequence mean abs logp err = {float(np.mean(seq_errs)):.6f}")
    print(f"  per-batch={['%.3f' % e for e in seq_errs]}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        main(d)

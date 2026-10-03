"""Autoregressive squad policy: sampling legality and log-probability parity.

The collector samples with ``forward`` and the learner re-scores the stored
action with ``evaluate_actions``. If those two disagree, Proximal Policy
Optimization optimises a distribution nobody sampled from, which shows up in
training as ``train/rollout_log_prob_abs_err`` far above ~1e-4 together with an
exploding ``train/approx_kl``. These tests pin the two together.
"""

import numpy as np
import pytest
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
    SQUAD_ACTION_COMMAND_IDX,
    SQUAD_ACTION_DIM,
    SQUAD_ACTION_SPATIAL_IDX,
    SQUAD_ACTION_TARGET_IDX,
    SQUAD_FLAT_ACTION_DIM,
    SQUAD_MASK_COMMAND_OFFSET,
    SQUAD_NULL_TARGET_SLOT,
    SQUAD_TARGET_SLOTS,
    squad_action_space_nvec,
    target_slot_kind,
)
from src.agent.universal_constants import (
    FIXED_UNIVERSAL_COMMAND_DIM,
    UNIVERSAL_DEFAULT_SPATIAL_IDX,
)

N_FEAT = 6
N_LAYERS = 3
HIDDEN = 32
# move_global (spatial), attack_unit (enemy), heal_unit (ally), load_unit (load)
COMMAND_MODES = [
    CMD_MODE_SPATIAL,
    CMD_MODE_ENEMY_TARGET,
    CMD_MODE_ALLY_TARGET,
    CMD_MODE_LOAD_TARGET,
]
N_CONFIGURED = len(COMMAND_MODES)


def _obs_space():
    return spaces.Dict(
        {
            "map": spaces.Box(0, 1, (N_LAYERS, 128, 128), dtype=np.float32),
            "command": spaces.Box(0, 1, (10,), dtype=np.float32),
            "anchor_tactical": spaces.Box(0, 1, (N_LAYERS, 32, 32), dtype=np.float32),
            "unit_scalars": spaces.Box(0, 1, (MAX_SQUAD_UNITS, N_FEAT), dtype=np.float32),
            "unit_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "enemy_scalars": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES, N_FEAT), dtype=np.float32),
            "enemy_mask": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES,), dtype=np.float32),
            "command_modes": spaces.Box(0, 10, (FIXED_UNIVERSAL_COMMAND_DIM,), dtype=np.float32),
            "heal_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "heal_target_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "load_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "load_target_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "attack_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "aoe_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "snipe_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "unload_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "cloak_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "uncloak_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "move_actor_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), dtype=np.float32),
            "tactical_enemy_mask": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES,), dtype=np.float32),
            "engage_enemy_mask": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES,), dtype=np.float32),
            "snipe_target_mask": spaces.Box(0, 1, (MAX_SQUAD_ENEMIES,), dtype=np.float32),
        }
    )


def _make_policy():
    policy = SquadRecurrentPolicy(
        _obs_space(),
        spaces.MultiDiscrete(squad_action_space_nvec()),
        get_schedule_fn(3e-4),
        rnn_type="gru",
        lstm_hidden_size=HIDDEN,
        features_extractor_kwargs={"features_dim": HIDDEN, "thinking_steps": 1},
        net_arch=[],
    )
    policy.set_training_mode(False)
    return policy


def _obs(
    batch=4,
    *,
    n_alive=3,
    n_enemies=2,
    healer_slots=(0,),
    wounded_slots=(1,),
    load_actor_slots=(),
    load_target_slots=(),
    seed=0,
):
    rng = np.random.default_rng(seed)
    unit_mask = np.zeros((batch, MAX_SQUAD_UNITS), dtype=np.float32)
    unit_mask[:, :n_alive] = 1.0
    enemy_mask = np.zeros((batch, MAX_SQUAD_ENEMIES), dtype=np.float32)
    enemy_mask[:, :n_enemies] = 1.0
    heal_actor = np.zeros((batch, MAX_SQUAD_UNITS), dtype=np.float32)
    for slot in healer_slots:
        heal_actor[:, slot] = 1.0
    heal_target = np.zeros((batch, MAX_SQUAD_UNITS), dtype=np.float32)
    for slot in wounded_slots:
        heal_target[:, slot] = 1.0
    load_actor = np.zeros((batch, MAX_SQUAD_UNITS), dtype=np.float32)
    for slot in load_actor_slots:
        load_actor[:, slot] = 1.0
    load_target = np.zeros((batch, MAX_SQUAD_UNITS), dtype=np.float32)
    for slot in load_target_slots:
        load_target[:, slot] = 1.0
    modes = np.full(
        (batch, FIXED_UNIVERSAL_COMMAND_DIM), float(CMD_MODE_NONE), dtype=np.float32
    )
    for idx, mode in enumerate(COMMAND_MODES):
        modes[:, idx] = float(mode)
    return {
        "map": rng.random((batch, N_LAYERS, 128, 128), dtype=np.float32),
        "command": rng.random((batch, 10), dtype=np.float32),
        "anchor_tactical": rng.random((batch, N_LAYERS, 32, 32), dtype=np.float32),
        "unit_scalars": rng.random((batch, MAX_SQUAD_UNITS, N_FEAT), dtype=np.float32)
        * unit_mask[..., None],
        "unit_mask": unit_mask,
        "enemy_scalars": rng.random((batch, MAX_SQUAD_ENEMIES, N_FEAT), dtype=np.float32)
        * enemy_mask[..., None],
        "enemy_mask": enemy_mask,
        "command_modes": modes,
        "heal_actor_mask": heal_actor,
        "heal_target_mask": heal_target,
        "load_actor_mask": load_actor,
        "load_target_mask": load_target,
        "attack_actor_mask": unit_mask.copy(),
        "aoe_actor_mask": unit_mask.copy(),
        "snipe_actor_mask": unit_mask.copy(),
        "unload_actor_mask": np.zeros_like(unit_mask),
        "cloak_actor_mask": np.zeros_like(unit_mask),
        "uncloak_actor_mask": np.zeros_like(unit_mask),
        "move_actor_mask": unit_mask.copy(),
        "tactical_enemy_mask": enemy_mask.copy(),
        "engage_enemy_mask": enemy_mask.copy(),
        "snipe_target_mask": enemy_mask.copy(),
    }


def _flat_mask(batch=4, *, n_alive=3, legal_commands=range(N_CONFIGURED)):
    mask = np.ones((batch, SQUAD_FLAT_ACTION_DIM), dtype=bool)
    for slot in range(n_alive, MAX_SQUAD_UNITS):
        mask[:, slot * 2] = True
        mask[:, slot * 2 + 1] = False
    cmd = SQUAD_MASK_COMMAND_OFFSET
    mask[:, cmd : cmd + FIXED_UNIVERSAL_COMMAND_DIM] = False
    for idx in legal_commands:
        mask[:, cmd + idx] = True
    return mask


def _zero_states(policy, batch):
    h = torch.zeros(1, batch, HIDDEN)
    return RNNStates((h,), (h,))


def _run(policy, obs, mask, batch, *, deterministic=False):
    obs_t = policy.obs_to_tensor(obs)[0]
    return policy.forward(
        obs_t,
        _zero_states(policy, batch),
        torch.ones(batch),
        action_masks=torch.as_tensor(mask),
        deterministic=deterministic,
    )


def test_action_vector_shape_matches_action_space():
    batch = 4
    policy = _make_policy()
    with torch.no_grad():
        actions, values, log_prob, _ = _run(policy, _obs(batch), _flat_mask(batch), batch)
    assert actions.shape == (batch, SQUAD_ACTION_DIM)
    assert values.shape == (batch, 1)
    assert log_prob.shape == (batch,)
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_forward_log_prob_matches_evaluate_actions(seed):
    """Collector and learner must score the same action identically."""
    batch = 6
    policy = _make_policy()
    obs = _obs(batch, seed=seed)
    mask = _flat_mask(batch)
    obs_t = policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        actions, _, log_prob, _ = _run(policy, obs, mask, batch)
        _, eval_log_prob, entropy = policy.evaluate_actions(
            obs_t,
            actions,
            _zero_states(policy, batch),
            torch.ones(batch),
            action_masks=torch.as_tensor(mask),
        )
    err = (eval_log_prob - log_prob).abs().max().item()
    assert err < 1e-4, f"max |log_p_learner - log_p_collector| = {err}"
    assert torch.isfinite(entropy).all()
    assert (entropy >= -1e-5).all()


def test_evaluate_actions_survives_drifted_command_target_masks():
    """Learner remask must not assign a huge penalty to a collected unit-target.

    If ``command_modes`` / heal target bits are missing (zeros) in a buffer row,
    categorical remasking used to score the stored ally slot as illegal
    (``_MASK_NEG`` ≈ -1e8 historically). Taken-action unmasking plus alive-slot
    actor weighting keep the abs error in a normal PPO range.
    """
    from src.agent.models.squad_policy import _MASK_NEG

    batch = 8
    policy = _make_policy()
    obs = _obs(batch, n_alive=4, healer_slots=(2,), wounded_slots=(3,), seed=41)
    mask = _flat_mask(batch, n_alive=4, legal_commands=[2])  # heal_unit only
    obs_t = policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        actions, _, log_prob, _ = _run(policy, obs, mask, batch)
        bad = {k: v.clone() if torch.is_tensor(v) else v for k, v in obs_t.items()}
        bad["command_modes"] = torch.zeros_like(bad["command_modes"])
        bad["heal_actor_mask"] = torch.zeros_like(bad["heal_actor_mask"])
        bad["heal_target_mask"] = torch.zeros_like(bad["heal_target_mask"])
        _, eval_log_prob, _ = policy.evaluate_actions(
            bad,
            actions,
            _zero_states(policy, batch),
            torch.ones(batch),
            action_masks=torch.as_tensor(mask),
        )
    assert torch.isfinite(eval_log_prob).all()
    assert (eval_log_prob > _MASK_NEG + 1.0).all()
    err = (eval_log_prob - log_prob).abs().mean().item()
    # Softmax renormalisation under a wiped target mask can move a few nats;
    # must stay nowhere near the old 1e7–1e8 abs-err regime.
    assert err < 20.0, f"mean abs logp err under drifted masks = {err}"
    assert err < abs(_MASK_NEG) / 2.0


def test_heal_actor_weight_uses_alive_not_heal_mask_alone():
    """Collect and evaluate must agree when heal_actor_mask is present.

    Regression for the multi-nat ``rollout_log_prob_abs_err`` after the
    autoregressive heal refactor: both paths must score the same actor bits.
    """
    batch = 8
    policy = _make_policy()
    obs = _obs(batch, n_alive=4, healer_slots=(2,), wounded_slots=(3,), seed=7)
    mask = _flat_mask(batch, n_alive=4, legal_commands=[2])
    obs_t = policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        actions, _, log_prob, _ = _run(policy, obs, mask, batch)
        _, eval_log_prob, _ = policy.evaluate_actions(
            obs_t,
            actions,
            _zero_states(policy, batch),
            torch.ones(batch),
            action_masks=torch.as_tensor(mask),
        )
    err = (eval_log_prob - log_prob).abs().max().item()
    assert err < 1e-4, f"heal collect/eval logπ drift = {err}"


def test_forbidden_alive_slots_do_not_dominate_actor_log_prob():
    """Units the command forces off must not add log(1-sigmoid).

    A strong pointer on an alive but ineligible slot used to contribute about
    -20 per unit. The stored log-prob fell near -150, and the collector and
    learner disagreed by a few nats, which skipped every update.
    """
    batch = 4
    n_alive = 4
    policy = _make_policy()
    obs = _obs(batch, n_alive=n_alive, healer_slots=(0,), wounded_slots=(1,), seed=3)
    mask = _flat_mask(batch, n_alive=n_alive, legal_commands=[2])
    real_forward = policy.features_extractor.forward

    def _strong_pointer(observations):
        out = real_forward(observations)
        n = observations["map"].shape[0]
        policy.features_extractor._pointer_logits = torch.full(
            (n, MAX_SQUAD_UNITS), 20.0, device=out.device
        )
        return out

    policy.features_extractor.forward = _strong_pointer
    obs_t = policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        actions, _, log_prob, _ = _run(policy, obs, mask, batch)
        _, eval_log_prob, _ = policy.evaluate_actions(
            obs_t,
            actions,
            _zero_states(policy, batch),
            torch.ones(batch),
            action_masks=torch.as_tensor(mask),
        )
    # Only the healer slot may be selected. The other alive slots are forced off.
    assert (actions[:, 1:n_alive] == 0).all()
    err = (eval_log_prob - log_prob).abs().max().item()
    assert err < 1e-4, f"forbidden-slot collect/eval drift = {err}"
    # One stochastic slot at logit 20 is at worst ~-20. Three forced slots
    # would add another ~-60 if they were still charged.
    assert float(log_prob.min()) > -40.0, f"actor logπ collapsed to {float(log_prob.min())}"


def test_spatial_command_forces_null_target_and_free_cell():
    batch = 8
    policy = _make_policy()
    obs = _obs(batch, seed=7)
    mask = _flat_mask(batch, legal_commands=[0])  # move_global only
    with torch.no_grad():
        actions, _, _, _ = _run(policy, obs, mask, batch)
    assert (actions[:, SQUAD_ACTION_COMMAND_IDX] == 0).all()
    assert (actions[:, SQUAD_ACTION_TARGET_IDX] == SQUAD_NULL_TARGET_SLOT).all()


def test_enemy_pointer_command_targets_visible_enemy_slot_and_pins_cell():
    batch = 8
    policy = _make_policy()
    obs = _obs(batch, n_enemies=2, seed=11)
    mask = _flat_mask(batch, legal_commands=[1])  # attack_unit only
    with torch.no_grad():
        actions, _, _, _ = _run(policy, obs, mask, batch)
    slots = actions[:, SQUAD_ACTION_TARGET_IDX].tolist()
    assert all(target_slot_kind(s) == CMD_MODE_ENEMY_TARGET for s in slots)
    assert all(0 <= s - MAX_SQUAD_UNITS < 2 for s in slots)
    # Non-spatial command: the world cell is pinned, so it adds no probability.
    assert (actions[:, SQUAD_ACTION_SPATIAL_IDX] == UNIVERSAL_DEFAULT_SPATIAL_IDX).all()


def test_ally_pointer_command_only_targets_legal_patients_and_ready_healers():
    batch = 8
    policy = _make_policy()
    obs = _obs(batch, n_alive=4, healer_slots=(2,), wounded_slots=(3,), seed=13)
    mask = _flat_mask(batch, n_alive=4, legal_commands=[2])  # heal_unit only
    with torch.no_grad():
        actions, _, _, _ = _run(policy, obs, mask, batch)
    assert (actions[:, SQUAD_ACTION_TARGET_IDX] == 3).all()
    # Only the ready healer may be an actor for heal_unit.
    actor_bits = actions[:, :MAX_SQUAD_UNITS]
    assert (actor_bits[:, [0, 1, 3]] == 0).all()


def test_load_pointer_command_only_targets_legal_partners_and_loaders():
    batch = 8
    policy = _make_policy()
    obs = _obs(
        batch,
        n_alive=4,
        healer_slots=(),
        wounded_slots=(),
        load_actor_slots=(0,),
        load_target_slots=(1,),
        seed=29,
    )
    mask = _flat_mask(batch, n_alive=4, legal_commands=[3])  # load_unit only
    with torch.no_grad():
        actions, _, _, _ = _run(policy, obs, mask, batch)
    assert (actions[:, SQUAD_ACTION_TARGET_IDX] == 1).all()
    actor_bits = actions[:, :MAX_SQUAD_UNITS]
    assert (actor_bits[:, [1, 2, 3]] == 0).all()
    assert (actions[:, SQUAD_ACTION_SPATIAL_IDX] == UNIVERSAL_DEFAULT_SPATIAL_IDX).all()


def test_unit_pointer_command_with_no_legal_target_falls_back_to_null():
    batch = 4
    policy = _make_policy()
    obs = _obs(batch, n_enemies=0, seed=17)
    mask = _flat_mask(batch, legal_commands=[1])  # attack_unit, but no enemies
    with torch.no_grad():
        actions, _, log_prob, _ = _run(policy, obs, mask, batch)
    assert (actions[:, SQUAD_ACTION_TARGET_IDX] == SQUAD_NULL_TARGET_SLOT).all()
    assert torch.isfinite(log_prob).all()


def test_dead_slots_are_never_selected_as_actors():
    batch = 8
    policy = _make_policy()
    n_alive = 2
    obs = _obs(batch, n_alive=n_alive, seed=19)
    mask = _flat_mask(batch, n_alive=n_alive)
    with torch.no_grad():
        actions, _, _, _ = _run(policy, obs, mask, batch)
    assert (actions[:, n_alive:MAX_SQUAD_UNITS] == 0).all()


def test_masked_command_slots_are_never_sampled():
    batch = 16
    policy = _make_policy()
    obs = _obs(batch, seed=23)
    mask = _flat_mask(batch, legal_commands=[0, 2])
    with torch.no_grad():
        actions, _, _, _ = _run(policy, obs, mask, batch)
    sampled = set(actions[:, SQUAD_ACTION_COMMAND_IDX].tolist())
    assert sampled <= {0, 2}


def test_target_slot_axis_covers_allies_enemies_and_null():
    nvec = squad_action_space_nvec()
    assert nvec[SQUAD_ACTION_TARGET_IDX] == SQUAD_TARGET_SLOTS
    assert SQUAD_TARGET_SLOTS == MAX_SQUAD_UNITS + MAX_SQUAD_ENEMIES + 1

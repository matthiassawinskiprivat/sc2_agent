"""Relational decode extras stay identity at zero init and condition later factors."""

import torch

from tests.test_squad_autoregressive_policy import (
    HIDDEN,
    SQUAD_NULL_TARGET_SLOT,
    _flat_mask,
    _make_policy,
    _obs,
    _run,
    _zero_states,
)


def _make_relational_policy():
    from gymnasium import spaces
    from stable_baselines3.common.utils import get_schedule_fn

    from src.agent.models.squad_policy import SquadRecurrentPolicy
    from src.agent.squad_constants import squad_action_space_nvec
    from tests.test_squad_autoregressive_policy import _obs_space

    policy = SquadRecurrentPolicy(
        _obs_space(),
        spaces.MultiDiscrete(squad_action_space_nvec()),
        get_schedule_fn(3e-4),
        rnn_type="gru",
        lstm_hidden_size=HIDDEN,
        features_extractor_kwargs={"features_dim": HIDDEN, "thinking_steps": 1},
        use_relational_decode=True,
        net_arch=[],
    )
    policy.set_training_mode(False)
    return policy


def _copy_overlap(src, dst):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    overlap = {
        k: v
        for k, v in src_sd.items()
        if k in dst_sd and dst_sd[k].shape == v.shape
    }
    dst.load_state_dict(overlap, strict=False)


def test_flag_off_has_no_relational_modules():
    off = _make_policy()
    assert off.use_relational_decode is False
    assert not hasattr(off, "actor_cmd_attn_scale")
    assert not hasattr(off, "target_selection_adapter")
    assert off.features_extractor.side_cross_attention is None


def test_flag_on_creates_zero_init_modules():
    on = _make_relational_policy()
    assert on.use_relational_decode is True
    assert float(on.actor_cmd_attn_scale.detach()) == 0.0
    assert torch.count_nonzero(on.target_selection_adapter.weight) == 0
    assert torch.count_nonzero(on.spatial_selection_adapter.weight) == 0
    assert on.features_extractor.side_cross_attention is not None
    assert torch.equal(
        on.features_extractor.side_cross_attention.out_proj.weight,
        torch.zeros_like(on.features_extractor.side_cross_attention.out_proj.weight),
    )


def test_zero_init_matches_policy_without_modules():
    batch = 4
    off = _make_policy()
    on = _make_relational_policy()
    _copy_overlap(off, on)
    obs = _obs(batch, seed=3)
    mask = _flat_mask(batch)
    with torch.no_grad():
        actions_off, _, lp_off, _ = _run(
            off, obs, mask, batch, deterministic=True
        )
        actions_on, _, lp_on, _ = _run(
            on, obs, mask, batch, deterministic=True
        )
        obs_t = on.obs_to_tensor(obs)[0]
        _, eval_on, _ = on.evaluate_actions(
            obs_t,
            actions_on,
            _zero_states(on, batch),
            torch.ones(batch),
            action_masks=torch.as_tensor(mask),
        )
    assert torch.equal(actions_off, actions_on)
    assert torch.allclose(lp_off, lp_on, atol=1e-5)
    assert torch.allclose(lp_on, eval_on, atol=1e-5)


def test_selection_changes_target_and_spatial_log_prob():
    from src.agent.models.squad_policy import _categorical_log_prob

    batch = 4
    policy = _make_relational_policy()
    with torch.no_grad():
        torch.nn.init.orthogonal_(policy.target_selection_adapter.weight, gain=2.0)
        torch.nn.init.orthogonal_(policy.spatial_selection_adapter.weight, gain=2.0)
        policy.target_selection_adapter.weight.mul_(20.0)
        policy.spatial_selection_adapter.weight.mul_(20.0)
    obs = _obs(batch, n_alive=4, seed=5)
    obs_t = policy.obs_to_tensor(obs)[0]
    bits_a = torch.zeros(batch, 24)
    bits_b = torch.zeros(batch, 24)
    bits_a[:, 1] = 1
    bits_b[:, 0] = 1
    with torch.no_grad():
        features = policy.extract_features(obs_t)
        l_pi, _ = policy._process_sequence(
            features, _zero_states(policy, batch).pi, torch.ones(batch), policy.lstm_actor
        )
        l_pi = policy.mlp_extractor.forward_actor(l_pi)
        ally = policy.features_extractor._ally_embeddings
        enemy = policy.features_extractor._enemy_embeddings
        fused = policy.features_extractor._spatial_fused
        ctx_a = policy._selected_actor_context(bits_a, ally, l_pi)
        ctx_b = policy._selected_actor_context(bits_b, ally, l_pi)
        assert not torch.allclose(ctx_a, ctx_b, atol=1e-5)
        cmd_embed = policy.command_embed(torch.zeros(batch, dtype=torch.long))
        target_a = policy.target_pointer(
            l_pi + policy.target_selection_adapter(ctx_a), cmd_embed, ally, enemy
        )
        target_b = policy.target_pointer(
            l_pi + policy.target_selection_adapter(ctx_b), cmd_embed, ally, enemy
        )
        spatial_a = policy.spatial_decoder(
            l_pi + policy.spatial_selection_adapter(ctx_a), fused
        )
        spatial_b = policy.spatial_decoder(
            l_pi + policy.spatial_selection_adapter(ctx_b), fused
        )
        targets = torch.full((batch,), SQUAD_NULL_TARGET_SLOT, dtype=torch.long)
        assert not torch.allclose(
            _categorical_log_prob(target_a, targets),
            _categorical_log_prob(target_b, targets),
            atol=1e-5,
        )
        # SpatialDecoder is ortho-inited at gain 0.01, so per-cell log-prob of a
        # near-uniform map barely moves; the logits themselves must still differ.
        assert not torch.allclose(spatial_a, spatial_b, atol=1e-8)


def test_forward_matches_evaluate_with_nonzero_relational_weights(seed=4):
    batch = 6
    policy = _make_relational_policy()
    with torch.no_grad():
        policy.actor_cmd_attn_scale.fill_(0.3)
        policy.target_selection_adapter.weight.fill_(0.02)
        policy.spatial_selection_adapter.weight.fill_(0.03)
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
    assert err < 1e-4, f"relational collect/eval logπ drift = {err}"
    assert torch.isfinite(entropy).all()

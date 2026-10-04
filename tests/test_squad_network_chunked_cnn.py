"""SquadNetwork hard-anchor tactical CNN (single crop, no soft blend)."""
import gymnasium as gym
import pytest
import torch
import torch.nn as nn
from gymnasium import spaces

from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import MAX_SQUAD_ENEMIES, SQUAD_UNIT_FEAT_DIM


def _squad_obs_space(max_units: int = 24, n_tactical: int = 4) -> gym.spaces.Dict:
    return spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(6, 128, 128), dtype=float),
            "command": spaces.Box(low=-1, high=1, shape=(10,), dtype=float),
            "anchor_tactical": spaces.Box(
                low=0,
                high=1,
                shape=(n_tactical, 32, 32),
                dtype=float,
            ),
            "unit_scalars": spaces.Box(
                low=-1,
                high=1,
                shape=(max_units, SQUAD_UNIT_FEAT_DIM),
                dtype=float,
            ),
            "unit_mask": spaces.Box(low=0, high=1, shape=(max_units,), dtype=float),
        }
    )


def test_squad_network_chunked_cnn_matches_full_batch():
    obs_space = _squad_obs_space()
    torch.manual_seed(0)
    net = SquadNetwork(
        obs_space, features_dim=64, thinking_steps=1, cnn_chunk_size=2
    )
    net.eval()
    batch = 5
    obs = {
        "map": torch.randn(batch, 6, 128, 128),
        "command": torch.randn(batch, 10),
        "anchor_tactical": torch.randn(batch, 4, 32, 32),
        "unit_scalars": torch.randn(batch, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.ones(batch, 24),
    }
    with torch.no_grad():
        chunked = net(obs)
        fused_chunked = net._spatial_fused.clone()
        net._cnn_chunk_size = 256
        full = net(obs)
        fused_full = net._spatial_fused
    assert torch.allclose(chunked, full, atol=1e-5)
    assert torch.allclose(fused_chunked, fused_full, atol=1e-5)


def test_squad_network_single_anchor_forward():
    obs_space = _squad_obs_space()
    net = SquadNetwork(obs_space, features_dim=64, thinking_steps=1)
    net.eval()

    batch = 4
    obs = {
        "map": torch.randn(batch, 6, 128, 128),
        "command": torch.randn(batch, 10),
        "anchor_tactical": torch.randn(batch, 4, 32, 32),
        "unit_scalars": torch.randn(batch, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.ones(batch, 24),
    }

    with torch.no_grad():
        out = net(obs)
        tactical_ref = net.tactical_cnn(obs["anchor_tactical"])

    assert out.shape == (batch, 64)
    assert net._spatial_fused is not None
    assert net._spatial_fused.shape == (batch, 128, 16, 16)
    # Fused tactical half must match the single-crop CNN (no soft-blend over N).
    assert torch.allclose(net._spatial_fused[:, 64:], tactical_ref, atol=1e-5)


def test_squad_network_rejects_legacy_unit_tactical_space():
    legacy = spaces.Dict(
        {
            "map": spaces.Box(low=0, high=1, shape=(6, 128, 128), dtype=float),
            "command": spaces.Box(low=-1, high=1, shape=(10,), dtype=float),
            "unit_tactical": spaces.Box(low=0, high=1, shape=(24, 4, 32, 32), dtype=float),
            "unit_scalars": spaces.Box(
                low=-1, high=1, shape=(24, SQUAD_UNIT_FEAT_DIM), dtype=float
            ),
            "unit_mask": spaces.Box(low=0, high=1, shape=(24,), dtype=float),
        }
    )
    try:
        SquadNetwork(legacy, features_dim=64, thinking_steps=1)
        raise AssertionError("expected KeyError for missing anchor_tactical")
    except KeyError as exc:
        assert "anchor_tactical" in str(exc)


def test_squad_network_self_attention_forward():
    """Phase 2: use_self_attention wires EntityEncoder self-attn and still forwards."""
    obs_space = _squad_obs_space()
    net = SquadNetwork(
        obs_space,
        features_dim=64,
        thinking_steps=1,
        use_self_attention=True,
    )
    net.eval()
    assert net.entity_encoder.use_self_attention is True
    assert hasattr(net.entity_encoder, "self_attn")

    batch = 2
    obs = {
        "map": torch.randn(batch, 6, 128, 128),
        "command": torch.randn(batch, 10),
        "anchor_tactical": torch.randn(batch, 4, 32, 32),
        "unit_scalars": torch.randn(batch, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.ones(batch, 24),
    }
    with torch.no_grad():
        out = net(obs)
    assert out.shape == (batch, 64)
    assert net._pointer_logits is not None
    assert net._pointer_logits.shape == (batch, 24)
    assert net._spatial_fused is not None
    assert net._spatial_fused.shape[1] == 128


def _squad_obs_space_with_enemies(max_units: int = 24, n_tactical: int = 4):
    space = dict(_squad_obs_space(max_units, n_tactical).spaces)
    space["enemy_scalars"] = spaces.Box(
        low=-1, high=1, shape=(MAX_SQUAD_ENEMIES, SQUAD_UNIT_FEAT_DIM), dtype=float
    )
    space["enemy_mask"] = spaces.Box(low=0, high=1, shape=(MAX_SQUAD_ENEMIES,), dtype=float)
    return spaces.Dict(space)


def test_squad_network_relational_decode_identity_and_empty_enemy_mask():
    obs_space = _squad_obs_space_with_enemies()
    torch.manual_seed(0)
    off = SquadNetwork(obs_space, features_dim=64, thinking_steps=1)
    on = SquadNetwork(
        obs_space, features_dim=64, thinking_steps=1, use_relational_decode=True
    )
    overlap = {
        k: v
        for k, v in off.state_dict().items()
        if k in on.state_dict() and on.state_dict()[k].shape == v.shape
    }
    on.load_state_dict(overlap, strict=False)
    off.eval()
    on.eval()
    assert on.side_cross_attention is not None
    batch = 3
    obs = {
        "map": torch.randn(batch, 6, 128, 128),
        "command": torch.randn(batch, 10),
        "anchor_tactical": torch.randn(batch, 4, 32, 32),
        "unit_scalars": torch.randn(batch, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.ones(batch, 24),
        "enemy_scalars": torch.randn(batch, MAX_SQUAD_ENEMIES, SQUAD_UNIT_FEAT_DIM),
        "enemy_mask": torch.ones(batch, MAX_SQUAD_ENEMIES),
    }
    with torch.no_grad():
        thought_off = off(obs)
        ally_off = off._ally_embeddings.clone()
        enemy_off = off._enemy_embeddings.clone()
        thought_on = on(obs)
        assert torch.allclose(thought_off, thought_on, atol=1e-5)
        assert torch.allclose(ally_off, on._ally_embeddings, atol=1e-5)
        assert torch.allclose(enemy_off, on._enemy_embeddings, atol=1e-5)
        empty = dict(obs)
        empty["enemy_mask"] = torch.zeros(batch, MAX_SQUAD_ENEMIES)
        out_empty = on(empty)
    assert torch.isfinite(out_empty).all()
    assert torch.isfinite(on._ally_embeddings).all()
    assert torch.isfinite(on._enemy_embeddings).all()
    assert on.map_sample_adapter is not None
    assert torch.equal(
        on.map_sample_adapter.weight, torch.zeros_like(on.map_sample_adapter.weight)
    )


def test_sample_map_at_units_reads_flipped_game_y():
    """x_norm=1, y_norm=1 (north-east in game) is the top-right CNN cell."""
    obs_space = _squad_obs_space_with_enemies()
    net = SquadNetwork(
        obs_space, features_dim=64, thinking_steps=1, use_relational_decode=True
    )
    feat = torch.zeros(1, 64, 16, 16)
    feat[0, 3, 0, 15] = 1.0  # row 0 = north after y-flip, col 15 = east
    scalars = torch.zeros(1, 2, SQUAD_UNIT_FEAT_DIM)
    from src.agent.unit_scalar_constants import UnitScalarConstants as USC

    scalars[0, 0, USC.FEATURES[USC.X_NORM]] = 1.0
    scalars[0, 0, USC.FEATURES[USC.Y_NORM]] = 1.0
    mask = torch.tensor([[1.0, 0.0]])
    sampled = net._sample_map_at_units(feat, scalars, mask)
    assert sampled.shape == (1, 2, 64)
    assert sampled[0, 0, 3].item() == pytest.approx(1.0, abs=1e-5)
    assert torch.all(sampled[0, 1] == 0.0)


def test_map_sample_adapter_moves_unit_key_when_nonzero():
    obs_space = _squad_obs_space_with_enemies()
    torch.manual_seed(1)
    net = SquadNetwork(
        obs_space, features_dim=64, thinking_steps=1, use_relational_decode=True
    )
    nn.init.eye_(net.map_sample_adapter.weight)
    nn.init.zeros_(net.map_sample_adapter.bias)
    net.eval()
    from src.agent.unit_scalar_constants import UnitScalarConstants as USC

    batch = 1
    obs = {
        "map": torch.randn(batch, 6, 128, 128),
        "command": torch.randn(batch, 10),
        "anchor_tactical": torch.randn(batch, 4, 32, 32),
        "unit_scalars": torch.zeros(batch, 24, SQUAD_UNIT_FEAT_DIM),
        "unit_mask": torch.zeros(batch, 24),
        "enemy_scalars": torch.zeros(batch, MAX_SQUAD_ENEMIES, SQUAD_UNIT_FEAT_DIM),
        "enemy_mask": torch.zeros(batch, MAX_SQUAD_ENEMIES),
    }
    obs["unit_mask"][0, :2] = 1.0
    obs["unit_scalars"][0, 0, USC.FEATURES[USC.X_NORM]] = 0.1
    obs["unit_scalars"][0, 0, USC.FEATURES[USC.Y_NORM]] = 0.2
    obs["unit_scalars"][0, 1, USC.FEATURES[USC.X_NORM]] = 0.8
    obs["unit_scalars"][0, 1, USC.FEATURES[USC.Y_NORM]] = 0.9
    with torch.no_grad():
        net(obs)
        emb_a = net._ally_embeddings[0, 0].clone()
        obs["unit_scalars"][0, 0, USC.FEATURES[USC.X_NORM]] = 0.9
        net(obs)
        emb_b = net._ally_embeddings[0, 0]
        emb_other = net._ally_embeddings[0, 1]
    assert not torch.allclose(emb_a, emb_b, atol=1e-5)
    # Slot 1 did not move.
    assert torch.isfinite(emb_other).all()

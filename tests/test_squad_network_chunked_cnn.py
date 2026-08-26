"""SquadNetwork hard-anchor tactical CNN (single crop, no soft blend)."""
import gymnasium as gym
import torch
from gymnasium import spaces

from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import SQUAD_UNIT_FEAT_DIM


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

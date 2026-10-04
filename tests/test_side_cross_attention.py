"""SideCrossAttention: identity at zero init, transport, empty-mask safety."""

import torch
import torch.nn as nn

from src.models.components.side_cross_attention import SideCrossAttention


def test_zero_projection_leaves_embeddings_unchanged():
    torch.manual_seed(0)
    mod = SideCrossAttention(embed_dim=16, nhead=4)
    mod.eval()
    ally = torch.randn(3, 5, 16)
    enemy = torch.randn(3, 6, 16)
    ally_mask = torch.ones(3, 5)
    enemy_mask = torch.ones(3, 6)
    with torch.no_grad():
        out_a, out_e = mod(ally, enemy, ally_mask, enemy_mask)
    assert torch.allclose(out_a, ally, atol=1e-6)
    assert torch.allclose(out_e, enemy, atol=1e-6)


def test_nonzero_projection_transports_enemy_information():
    torch.manual_seed(1)
    mod = SideCrossAttention(embed_dim=16, nhead=4)
    nn.init.eye_(mod.out_proj.weight)
    nn.init.zeros_(mod.out_proj.bias)
    mod.eval()
    ally = torch.randn(2, 4, 16)
    enemy = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4)
    enemy2 = enemy.clone()
    enemy2[..., :4] = enemy2[..., :4] + torch.linspace(0.5, 2.0, 4)
    with torch.no_grad():
        out_a, _ = mod(ally, enemy, mask, mask)
        out_a2, _ = mod(ally, enemy2, mask, mask)
    assert not torch.allclose(out_a, out_a2, atol=1e-5)


def test_empty_enemy_mask_stays_finite():
    torch.manual_seed(2)
    mod = SideCrossAttention(embed_dim=16, nhead=4)
    nn.init.normal_(mod.out_proj.weight, std=0.5)
    mod.eval()
    ally = torch.randn(3, 5, 16)
    enemy = torch.randn(3, 5, 16)
    ally_mask = torch.ones(3, 5)
    enemy_mask = torch.zeros(3, 5)
    with torch.no_grad():
        out_a, out_e = mod(ally, enemy, ally_mask, enemy_mask)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_e).all()
    # No keys → skip that direction; ally stays put.
    assert torch.allclose(out_a, ally, atol=1e-6)


def test_fully_masked_batch_finite():
    mod = SideCrossAttention(embed_dim=16, nhead=4)
    nn.init.normal_(mod.out_proj.weight, std=0.25)
    mod.eval()
    ally = torch.randn(2, 4, 16)
    enemy = torch.randn(2, 4, 16)
    zeros = torch.zeros(2, 4)
    with torch.no_grad():
        out_a, out_e = mod(ally, enemy, zeros, zeros)
    assert torch.isfinite(out_a).all()
    assert torch.isfinite(out_e).all()
    assert torch.allclose(out_a, ally)
    assert torch.allclose(out_e, enemy)

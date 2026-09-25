"""Tests for EntityEncoder self-attention path."""
import torch
import pytest
from src.models.components.entity_encoder import EntityEncoder


def test_without_self_attention_output_shape():
    enc = EntityEncoder(input_dim=16, embed_dim=32, hidden_dim=32, use_self_attention=False)
    enc.eval()
    x = torch.randn(4, 8, 16)
    mask = torch.ones(4, 8)
    out = enc(x, mask)
    assert out.shape == (4, 8, 32)


def test_with_self_attention_output_shape():
    enc = EntityEncoder(input_dim=16, embed_dim=32, hidden_dim=32, use_self_attention=True)
    enc.eval()
    x = torch.randn(4, 8, 16)
    mask = torch.ones(4, 8)
    out = enc(x, mask)
    assert out.shape == (4, 8, 32)


def test_self_attention_module_created_only_when_requested():
    enc_no = EntityEncoder(input_dim=8, embed_dim=16, use_self_attention=False)
    assert not hasattr(enc_no, "self_attn")

    enc_yes = EntityEncoder(input_dim=8, embed_dim=16, use_self_attention=True)
    assert hasattr(enc_yes, "self_attn")


def test_padding_slots_zeroed_after_self_attention():
    """Units in padding slots must remain all-zero after self-attention."""
    enc = EntityEncoder(input_dim=8, embed_dim=16, hidden_dim=16, use_self_attention=True)
    enc.eval()
    batch, n_units = 2, 6
    x = torch.randn(batch, n_units, 8)
    # First 4 valid, last 2 padding
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        out = enc(x, mask)
    assert torch.all(out[:, 4:, :] == 0.0), "Padding slots must be zero after self-attention"


def test_self_attention_changes_output_vs_no_attention():
    """With self-attention enabled, output should differ from without (different weights)."""
    torch.manual_seed(42)
    enc_no = EntityEncoder(input_dim=8, embed_dim=16, hidden_dim=16, use_self_attention=False)
    enc_yes = EntityEncoder(input_dim=8, embed_dim=16, hidden_dim=16, use_self_attention=True)
    enc_no.eval()
    enc_yes.eval()
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4)
    with torch.no_grad():
        out_no = enc_no(x, mask)
        out_yes = enc_yes(x, mask)
    # Different weights → outputs won't be equal
    assert not torch.allclose(out_no, out_yes)


def test_no_mask_does_not_crash():
    enc = EntityEncoder(input_dim=8, embed_dim=16, use_self_attention=True)
    enc.eval()
    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        out = enc(x, mask=None)
    assert out.shape == (2, 4, 16)


def test_fully_masked_rows_finite_in_eval():
    """Empty ally/enemy sets must not emit NaN under eval (rollout collection)."""
    enc = EntityEncoder(input_dim=8, embed_dim=16, hidden_dim=16, use_self_attention=True)
    enc.eval()
    x = torch.randn(4, 6, 8)
    mask = torch.zeros(4, 6)
    mask[0, :2] = 1.0  # row 0 has entities; rows 1-3 fully padded
    with torch.no_grad():
        out = enc(x, mask)
    assert torch.isfinite(out).all()
    assert torch.all(out[1:] == 0.0)
    assert not torch.all(out[0, :2] == 0.0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_fully_masked_batch_finite_eval_cuda(device):
    enc = EntityEncoder(input_dim=16, embed_dim=32, hidden_dim=32, use_self_attention=True)
    enc.to(device).eval()
    x = torch.randn(8, 12, 16, device=device)
    mask = torch.zeros(8, 12, device=device)
    with torch.no_grad():
        out = enc(x, mask)
    assert torch.isfinite(out).all()
    assert torch.all(out == 0.0)

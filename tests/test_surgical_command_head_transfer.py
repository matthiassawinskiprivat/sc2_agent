"""
Curriculum load (_surgical_reconstruct_policy) copies the first N command_head.bias
entries from the checkpoint, then fills any new slots with the mean of the copied
values (avoids greedy argmax always preferring a trailing 0 vs negative old biases).

Distribution tests still document the old [-5,-5,0] failure mode for context.
"""

import torch
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution


def _surgical_merge_command_head_weight(
    checkpoint_weight: torch.Tensor, new_model_weight: torch.Tensor
) -> torch.Tensor:
    """Mirror maskable_recurrent_ppo command_head.weight expansion."""
    assert checkpoint_weight.dim() == 2 and new_model_weight.dim() == 2
    assert checkpoint_weight.shape[1] == new_model_weight.shape[1]
    assert checkpoint_weight.shape[0] <= new_model_weight.shape[0]
    exp = new_model_weight.clone()
    mi = checkpoint_weight.shape[0]
    exp[:mi, :] = checkpoint_weight
    if mi < exp.shape[0]:
        row_mean = checkpoint_weight.mean(dim=0, keepdim=True)
        exp[mi:, :] = row_mean.expand(exp.shape[0] - mi, -1)
    return exp


def test_surgical_command_head_weight_tail_is_row_mean():
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    c_sd = torch.zeros(3, 2, dtype=torch.float32)
    merged = _surgical_merge_command_head_weight(v, c_sd)
    mean_row = torch.tensor([0.5, 0.5], dtype=torch.float32)
    assert torch.allclose(merged[2], mean_row)


def _surgical_merge_command_head_bias(
    checkpoint_bias: torch.Tensor, new_model_bias: torch.Tensor
) -> torch.Tensor:
    """Mirror src/agent/maskable_recurrent_ppo.py::_surgical_reconstruct_policy bias branch."""
    assert checkpoint_bias.dim() == 1 and new_model_bias.dim() == 1
    assert checkpoint_bias.shape[0] <= new_model_bias.shape[0]
    exp = new_model_bias.clone()
    exp[: checkpoint_bias.shape[0]] = checkpoint_bias
    if checkpoint_bias.shape[0] < exp.shape[0]:
        exp[checkpoint_bias.shape[0] :] = checkpoint_bias.mean()
    return exp


def test_surgical_command_head_bias_tail_filled_with_mean_of_checkpoint():
    """Trailing command slots get mean(v), not left from clone."""
    v = torch.tensor([-3.0, -4.0], dtype=torch.float32)
    c_sd = torch.zeros(3, dtype=torch.float32)
    merged = _surgical_merge_command_head_bias(v, c_sd)
    assert torch.allclose(merged, torch.tensor([-3.0, -4.0, -3.5], dtype=torch.float32))


def test_surgical_command_head_bias_equal_prefix_tail_matches_prefix():
    """When all copied biases are equal, mean equals them — no stray zero tail."""
    v = torch.tensor([-5.0, -5.0], dtype=torch.float32)
    c_sd = torch.zeros(3, dtype=torch.float32)
    merged = _surgical_merge_command_head_bias(v, c_sd)
    assert torch.allclose(merged, torch.tensor([-5.0, -5.0, -5.0], dtype=torch.float32))


def test_greedy_after_mean_merge_does_not_force_last_index():
    """Biases [-5,-5,-5] tie; deterministic argmax picks first index 0."""
    n_cmd, n_spatial = 3, 8
    dist = MaskableMultiCategoricalDistribution([n_cmd, n_spatial])
    v = torch.tensor([-5.0, -5.0], dtype=torch.float32)
    c_sd = torch.zeros(3, dtype=torch.float32)
    cmd_bias = _surgical_merge_command_head_bias(v, c_sd)
    cmd = cmd_bias.unsqueeze(0)
    spatial = torch.zeros(1, n_spatial, dtype=torch.float32)
    logits = torch.cat([cmd, spatial], dim=1)
    dist.proba_distribution(logits)
    dist.apply_masking(torch.ones(1, n_cmd + n_spatial, dtype=torch.bool))
    actions = dist.get_actions(deterministic=True)
    assert int(actions[0, 0].item()) == 0


def test_greedy_multidiscrete_zero_third_bias_prefers_last_index_regression_doc():
    """
    Legacy failure mode: if the third bias stayed 0 while the first two were -5,
    greedy command selection is always index 2 (why we fill new slots with mean).
    """
    n_cmd, n_spatial = 3, 8
    dist = MaskableMultiCategoricalDistribution([n_cmd, n_spatial])
    cmd = torch.tensor([[-5.0, -5.0, 0.0]], dtype=torch.float32)
    spatial = torch.zeros(1, n_spatial, dtype=torch.float32)
    logits = torch.cat([cmd, spatial], dim=1)
    dist.proba_distribution(logits)
    dist.apply_masking(torch.ones(1, n_cmd + n_spatial, dtype=torch.bool))
    actions = dist.get_actions(deterministic=True)
    assert int(actions[0, 0].item()) == 2


def test_sampled_command_still_favors_index_two_with_negative_prefix_and_zero_tail():
    """Under softmax, index 2 has highest mass when third logit is 0 and others -5."""
    n_cmd, n_spatial = 3, 8
    dist = MaskableMultiCategoricalDistribution([n_cmd, n_spatial])
    cmd = torch.tensor([[-5.0, -5.0, 0.0]], dtype=torch.float32)
    spatial = torch.zeros(1, n_spatial, dtype=torch.float32)
    logits = torch.cat([cmd, spatial], dim=1)
    dist.proba_distribution(logits)
    dist.apply_masking(torch.ones(1, n_cmd + n_spatial, dtype=torch.bool))
    torch.manual_seed(42)
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(400):
        a = dist.get_actions(deterministic=False)
        counts[int(a[0, 0].item())] += 1
    assert counts[2] > counts[0] and counts[2] > counts[1]


def test_greedy_equal_command_logits_torch_argmax_first():
    """Ties on command logits: deterministic mode uses argmax → first index wins."""
    n_cmd, n_spatial = 3, 8
    dist = MaskableMultiCategoricalDistribution([n_cmd, n_spatial])
    cmd = torch.zeros(1, n_cmd, dtype=torch.float32)
    spatial = torch.zeros(1, n_spatial, dtype=torch.float32)
    logits = torch.cat([cmd, spatial], dim=1)
    dist.proba_distribution(logits)
    dist.apply_masking(torch.ones(1, n_cmd + n_spatial, dtype=torch.bool))
    actions = dist.get_actions(deterministic=True)
    assert int(actions[0, 0].item()) == 0


def test_surgical_merge_overwrites_tail_even_if_clone_was_nonzero():
    """New tail is always mean(checkpoint_prefix), not the pre-clone tail."""
    v = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    c_sd = torch.tensor([9.0, 9.0, 7.0], dtype=torch.float32)
    merged = _surgical_merge_command_head_bias(v, c_sd)
    assert torch.allclose(merged, torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32))

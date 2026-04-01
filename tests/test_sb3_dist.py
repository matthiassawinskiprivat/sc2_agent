import torch
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution


def test_masking_accepts_correct_shape():
    """Verify MaskableMultiCategoricalDistribution accepts a (batch, sum(nvec)) mask."""
    action_dims = [2, 1024]
    dist = MaskableMultiCategoricalDistribution(action_dims)

    logits = torch.zeros(4, 1026)
    dist.proba_distribution(action_logits=logits)

    mask = torch.ones((4, 1026), dtype=torch.bool)
    dist.apply_masking(mask)

    actions = dist.get_actions(deterministic=False)
    assert actions.shape == (4, 2), f"Expected (4, 2) actions, got {actions.shape}"
    assert (actions[:, 0] < 2).all(), "Command head action out of range"
    assert (actions[:, 1] < 1024).all(), "Spatial head action out of range"


def test_masking_blocks_action():
    """Verify that masking out a command head option prevents it from being sampled."""
    action_dims = [2, 1024]
    dist = MaskableMultiCategoricalDistribution(action_dims)

    logits = torch.zeros(1, 1026)
    dist.proba_distribution(action_logits=logits)

    mask = torch.ones((1, 1026), dtype=torch.bool)
    mask[0, 0] = False  # block command 0

    dist.apply_masking(mask)
    for _ in range(20):
        actions = dist.get_actions(deterministic=False)
        assert actions[0, 0].item() == 1, "Masked-out command 0 was still sampled"

import torch
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution

from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

_CMD_SP = FIXED_UNIVERSAL_COMMAND_DIM + 1024


def test_masking_accepts_correct_shape():
    """Verify MaskableMultiCategoricalDistribution accepts a (batch, sum(nvec)) mask."""
    action_dims = [FIXED_UNIVERSAL_COMMAND_DIM, 1024]
    dist = MaskableMultiCategoricalDistribution(action_dims)

    logits = torch.zeros(4, _CMD_SP)
    dist.proba_distribution(action_logits=logits)

    mask = torch.ones((4, _CMD_SP), dtype=torch.bool)
    dist.apply_masking(mask)

    actions = dist.get_actions(deterministic=False)
    assert actions.shape == (4, 2), f"Expected (4, 2) actions, got {actions.shape}"
    assert (actions[:, 0] < FIXED_UNIVERSAL_COMMAND_DIM).all(), "Command head action out of range"
    assert (actions[:, 1] < 1024).all(), "Spatial head action out of range"


def test_masking_blocks_action():
    """Only one command index is legal in the first head; sampling never picks another."""
    action_dims = [FIXED_UNIVERSAL_COMMAND_DIM, 1024]
    dist = MaskableMultiCategoricalDistribution(action_dims)

    logits = torch.zeros(1, _CMD_SP)
    dist.proba_distribution(action_logits=logits)

    mask = torch.zeros((1, _CMD_SP), dtype=torch.bool)
    mask[0, 1] = True  # first head: only command index 1 allowed
    mask[0, FIXED_UNIVERSAL_COMMAND_DIM :] = True  # spatial head unrestricted

    dist.apply_masking(mask)
    for _ in range(20):
        actions = dist.get_actions(deterministic=False)
        assert actions[0, 0].item() == 1, "Only command index 1 should be sampleable"

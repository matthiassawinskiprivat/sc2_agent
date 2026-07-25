"""Squad action mask: dead slots must force 'not selected', not both-False."""

import numpy as np


def test_squad_action_masks_dead_slots_force_not_selected():
    from src.agent.ctf_bot import CaptureTheFlagBot
    from src.agent.squad_constants import MAX_SQUAD_UNITS, squad_action_space_nvec
    from src.agent.universal_constants import FIXED_UNIVERSAL_COMMAND_DIM

    bot = CaptureTheFlagBot.__new__(CaptureTheFlagBot)
    bot.training_conf = {
        "action_heads": ["move_global", "move_local", "attack_local"],
        "categorical_heads": [],
    }

    unit_mask = np.zeros(MAX_SQUAD_UNITS, dtype=np.float32)
    unit_mask[:2] = 1.0
    mask = bot.squad_action_masks(unit_mask)

    assert mask.shape[0] == int(sum(squad_action_space_nvec()))
    # Alive slots: both binary options legal
    assert bool(mask[0]) and bool(mask[1])
    assert bool(mask[2]) and bool(mask[3])
    # Dead slots: only action 0 (not selected) legal
    for i in range(2, MAX_SQUAD_UNITS):
        assert bool(mask[i * 2]) is True
        assert bool(mask[i * 2 + 1]) is False

    cmd_start = MAX_SQUAD_UNITS * 2
    assert mask[cmd_start : cmd_start + 3].all()
    assert not mask[cmd_start + 3 : cmd_start + FIXED_UNIVERSAL_COMMAND_DIM].any()

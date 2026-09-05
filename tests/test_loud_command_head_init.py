"""One-time loud command prior at curriculum graft (heal_local)."""
import torch
import torch.nn as nn

from src.utils.model_utils import (
    DEFAULT_LOUD_COMMAND_BIAS,
    DEFAULT_LOUD_COMMAND_MIX,
    apply_loud_command_head_init,
    configured_command_names,
    loud_command_bias_from_conf,
    loud_command_heads_from_conf,
    loud_command_mix_from_conf,
    maybe_apply_loud_command_head_init,
)


COMMAND_NAMES = ["move_local", "attack_local", "heal_local"]


def _policy():
    policy = nn.Module()
    policy.command_head = nn.Linear(8, 64)
    with torch.no_grad():
        policy.command_head.weight.zero_()
        policy.command_head.bias.zero_()
        policy.command_head.weight[0].fill_(1.0)
        policy.command_head.weight[1].fill_(2.0)
        policy.command_head.bias[0] = 0.5
        policy.command_head.bias[1] = 1.25
        policy.command_head.weight[2].fill_(0.01)
    return policy


def test_loud_heads_from_nested_env():
    conf = {
        "env": {
            "loud_command_heads": ["heal_local"],
            "loud_command_mix": 0.5,
            "loud_command_bias": 0.0,
        }
    }
    assert loud_command_heads_from_conf(conf) == ["heal_local"]
    assert loud_command_mix_from_conf(conf) == 0.5
    assert loud_command_bias_from_conf(conf) == 0.0


def test_loud_mix_and_bias_defaults_when_omitted():
    assert loud_command_mix_from_conf({"env": {}}) == DEFAULT_LOUD_COMMAND_MIX
    assert loud_command_bias_from_conf({"env": {}}) == DEFAULT_LOUD_COMMAND_BIAS
    assert loud_command_heads_from_conf({}) == []


def test_configured_command_names_prefers_root_cumulative_list():
    conf = {
        "action_heads": ["move_local", "attack_local", "heal_local"],
        "categorical_heads": ["cloak_burrow"],
        "env": {"action_heads": ["heal_local"]},
    }
    assert configured_command_names(conf) == [
        "move_local",
        "attack_local",
        "heal_local",
        "cloak_burrow",
    ]


def test_heal_slot_full_copy_then_extra_bias():
    policy = _policy()
    boosted = apply_loud_command_head_init(
        policy, COMMAND_NAMES, ["heal_local"], extra_bias=1.0, mix=1.0
    )
    assert boosted == [2]
    assert torch.equal(policy.command_head.weight[2], policy.command_head.weight[1])
    assert abs(float(policy.command_head.bias[2]) - 2.25) < 1e-5
    assert abs(float(policy.command_head.bias[0]) - 0.5) < 1e-5
    assert abs(float(policy.command_head.bias[1]) - 1.25) < 1e-5
    assert torch.allclose(policy.command_head.weight[0], torch.ones(8))


def test_heal_slot_halfway_blend_no_extra_bias():
    policy = _policy()
    orig_w = policy.command_head.weight[2].detach().clone()
    orig_b = float(policy.command_head.bias[2])
    attack_w = policy.command_head.weight[1].detach().clone()
    boosted = apply_loud_command_head_init(
        policy, COMMAND_NAMES, ["heal_local"], extra_bias=0.0, mix=0.5
    )
    assert boosted == [2]
    expected_w = 0.5 * orig_w + 0.5 * attack_w
    assert torch.allclose(policy.command_head.weight[2], expected_w)
    assert abs(float(policy.command_head.bias[2]) - (0.5 * orig_b + 0.5 * 1.25)) < 1e-5
    assert torch.equal(policy.command_head.weight[1], attack_w)


def test_loud_init_is_idempotent():
    policy = _policy()
    apply_loud_command_head_init(
        policy, COMMAND_NAMES, ["heal_local"], extra_bias=1.0, mix=1.0
    )
    w = policy.command_head.weight[2].detach().clone()
    b = policy.command_head.bias[2].detach().clone()
    apply_loud_command_head_init(
        policy, COMMAND_NAMES, ["heal_local"], extra_bias=1.0, mix=1.0
    )
    assert torch.equal(policy.command_head.weight[2], w)
    assert torch.equal(policy.command_head.bias[2], b)


def test_missing_head_is_noop():
    policy = _policy()
    w_before = policy.command_head.weight.detach().clone()
    b_before = policy.command_head.bias.detach().clone()
    boosted = apply_loud_command_head_init(
        policy, COMMAND_NAMES, ["aoe_local"], extra_bias=1.0
    )
    assert boosted == []
    assert torch.equal(policy.command_head.weight, w_before)
    assert torch.equal(policy.command_head.bias, b_before)


def test_resume_skips_loud_init():
    policy = _policy()
    w_before = policy.command_head.weight.detach().clone()
    conf = {
        "action_heads": COMMAND_NAMES,
        "env": {"loud_command_heads": ["heal_local"], "loud_command_bias": 1.0},
    }
    assert maybe_apply_loud_command_head_init(policy, conf, resume=True) == []
    assert torch.equal(policy.command_head.weight, w_before)


def test_maybe_apply_reads_env_and_wraps_model():
    policy = _policy()

    class Model:
        pass

    model = Model()
    model.policy = policy
    conf = {
        "action_heads": COMMAND_NAMES,
        "env": {
            "loud_command_heads": ["HEAL_LOCAL"],
            "loud_command_mix": 1.0,
            "loud_command_bias": 0.5,
        },
    }
    boosted = maybe_apply_loud_command_head_init(model, conf, resume=False)
    assert boosted == [2]
    assert torch.equal(policy.command_head.weight[2], policy.command_head.weight[1])
    assert abs(float(policy.command_head.bias[2]) - 1.75) < 1e-5

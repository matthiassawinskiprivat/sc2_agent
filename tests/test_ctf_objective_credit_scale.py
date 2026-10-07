"""Objective-distance falloff for enemy-value credit."""
from sc2.position import Point2

from src.agent.ctf_economic_rewards import (
    ROLE_ATTACK,
    UnitValue,
    objective_credit_scale,
    value_delta,
)


def test_objective_credit_scale_full_inside_inner_radius():
    assert objective_credit_scale(
        9.0, inner_radius=10.0, falloff=15.0, min_scale=0.25
    ) == 1.0


def test_objective_credit_scale_hits_floor_after_falloff():
    assert objective_credit_scale(
        25.0, inner_radius=10.0, falloff=15.0, min_scale=0.25
    ) == 0.25
    assert objective_credit_scale(
        40.0, inner_radius=10.0, falloff=15.0, min_scale=0.25
    ) == 0.25


def test_objective_credit_scale_linear_mid_falloff():
    mid = objective_credit_scale(
        17.5, inner_radius=10.0, falloff=15.0, min_scale=0.25
    )
    assert abs(mid - 0.625) < 1e-9


def _uv(tag_pos, vitality=100.0, value=1.0):
    return UnitValue(
        type_id=1,
        value=value,
        max_vitality=100.0,
        vitality=vitality,
        position=tag_pos,
        role=ROLE_ATTACK,
    )


def test_value_delta_applies_position_scale_to_damage_not_restore():
    near = Point2((0.0, 0.0))
    far = Point2((40.0, 0.0))
    prev = {
        1: _uv(near, vitality=100.0),
        2: _uv(far, vitality=100.0),
    }
    curr = {
        1: _uv(near, vitality=50.0),
        2: _uv(far, vitality=50.0),
    }

    def scale(pos):
        if pos is None:
            return 1.0
        return 1.0 if pos.x < 10.0 else 0.25

    flow = value_delta(prev, curr, position_scale=scale)
    assert abs(flow.damaged - (0.5 * 1.0 + 0.5 * 0.25)) < 1e-9
    assert flow.destroyed == 0.0

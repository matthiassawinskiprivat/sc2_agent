"""Legacy CTF reward smoke tests.

Flat per-HP pylon coefficients were replaced by spawn-army-scaled pylon credit
(``pylon_value_scale``). See ``tests/test_ctf_pylon_value_scale.py``.
"""
import pytest


def test_ctf_reward_calculation():
    pytest.skip("replaced by spawn-army-scaled pylon credit in test_ctf_pylon_value_scale")


def test_ctf_unit_damage_reward():
    """Flat HP damage_reward was removed; see test_ctf_economic_death_reward /
    test_ctf_god_mode_combat_reward for the resource-weighted replacement."""
    pytest.skip("replaced by economic value combat rewards")


def test_melee_squad_damage_reward_and_penalty_scaling():
    pytest.skip("replaced by economic value combat rewards")


def test_ranged_squad_uses_base_damage_coefficients():
    pytest.skip("replaced by economic value combat rewards")

"""Upgrade-id filtering before GameState (python-sc2 / SC2 version skew)."""

from types import SimpleNamespace

import pytest

from sc2.ids.upgrade_id import UpgradeId


def _make_response_observation(upgrade_ids: list):
    player = SimpleNamespace(upgrade_ids=list(upgrade_ids))
    raw_data = SimpleNamespace(player=player)
    observation = SimpleNamespace(raw_data=raw_data)
    return SimpleNamespace(observation=observation)


def test_strip_unknown_upgrade_ids_removes_invalid():
    from src.utils.sc2_observation_compat import strip_unknown_upgrade_ids_from_response_observation

    any_valid = next(iter(UpgradeId)).value
    ro = _make_response_observation([any_valid, 305, 99999])
    strip_unknown_upgrade_ids_from_response_observation(ro)
    assert ro.observation.raw_data.player.upgrade_ids == [any_valid]


def test_strip_unknown_upgrade_ids_empty_player():
    from src.utils.sc2_observation_compat import strip_unknown_upgrade_ids_from_response_observation

    ro = SimpleNamespace(observation=SimpleNamespace())
    strip_unknown_upgrade_ids_from_response_observation(ro)  # no crash


def test_game_state_from_response_observation_strips_before_game_state():
    from unittest.mock import patch

    from src.utils.sc2_observation_compat import game_state_from_response_observation

    any_valid = next(iter(UpgradeId)).value
    ro = _make_response_observation([any_valid, 305])

    captured = {}

    class FakeGS:
        def __init__(self, response_observation, previous_observation=None):
            captured["ids"] = list(response_observation.observation.raw_data.player.upgrade_ids)

    with patch("sc2.game_state.GameState", FakeGS):
        game_state_from_response_observation(ro)

    assert captured["ids"] == [any_valid]

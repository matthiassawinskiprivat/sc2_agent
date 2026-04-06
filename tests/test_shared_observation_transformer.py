"""Tests for SharedObservationTransformer — caching, dict vs ndarray paths."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_shared_transformer(arch="universal", active_layers=None):
    """Build a SharedObservationTransformer with a mocked inner transformer."""
    from src.transformers.shared_observation_transformer import SharedObservationTransformer

    active_layers = active_layers or [
        "allied_unit_density", "enemy_unit_density",
        "terrain_pathable", "command_objective",
    ]
    training_conf = {
        "agent_architecture": arch,
        "active_layers": active_layers,
        "active_tactical_layers": active_layers,
    }

    with patch("src.transformers.shared_observation_transformer.ObservationTransformer") as MockOT:
        mock_inner = MockOT.return_value
        n = len(active_layers)

        mock_inner.transform.return_value = {
            "map": np.random.rand(n, 128, 128).astype(np.float32),
            "command": np.zeros(10, dtype=np.float32),
        }
        mock_inner.transform_tactical.return_value = np.random.rand(n, 32, 32).astype(np.float32)

        st = SharedObservationTransformer(
            map_size=(128, 128), training_conf=training_conf,
        )
    return st, mock_inner


class TestUniversalDictPath:
    def test_returns_dict_with_expected_keys(self):
        st, _ = _make_shared_transformer(arch="universal")
        result = st.transform_observation({"dummy": 1}, observation_id="obs_0")

        assert isinstance(result, dict)
        assert set(result.keys()) == {"map", "tactical", "command"}

    def test_map_shape(self):
        st, _ = _make_shared_transformer(arch="universal")
        result = st.transform_observation({"dummy": 1}, observation_id="obs_0")
        assert result["map"].shape == (4, 128, 128)

    def test_tactical_shape(self):
        st, _ = _make_shared_transformer(arch="universal")
        result = st.transform_observation({"dummy": 1}, observation_id="obs_0")
        assert result["tactical"].shape == (4, 32, 32)

    def test_command_shape(self):
        st, _ = _make_shared_transformer(arch="universal")
        result = st.transform_observation({"dummy": 1}, observation_id="obs_0")
        assert result["command"].shape == (10,)

    def test_tactical_receives_active_unit(self):
        st, mock_inner = _make_shared_transformer(arch="universal")
        mock_unit = MagicMock()
        st.transform_observation({"dummy": 1}, observation_id="obs_0", active_unit=mock_unit)

        mock_inner.transform_tactical.assert_called_once()
        _, kwargs = mock_inner.transform_tactical.call_args
        positional = mock_inner.transform_tactical.call_args[0]
        assert positional[1] is mock_unit


class TestCaching:
    def test_cache_hit_returns_same_object(self):
        st, mock_inner = _make_shared_transformer()
        u = MagicMock()
        u.tag = 1
        r1 = st.transform_observation({"a": 1}, observation_id="obs_1", active_unit=u)
        r2 = st.transform_observation({"a": 1}, observation_id="obs_1", active_unit=u)

        assert r1 is r2
        assert mock_inner.transform.call_count == 1

    def test_cache_miss_when_same_id_but_active_unit_changes(self):
        st, mock_inner = _make_shared_transformer()
        u1 = MagicMock()
        u1.tag = 1
        u2 = MagicMock()
        u2.tag = 2
        st.transform_observation({"a": 1}, observation_id="obs_1", active_unit=u1)
        st.transform_observation({"a": 1}, observation_id="obs_1", active_unit=u2)

        assert mock_inner.transform.call_count == 2
        assert mock_inner.transform_tactical.call_count == 2

    def test_cache_miss_on_new_observation_id(self):
        st, mock_inner = _make_shared_transformer()
        st.transform_observation({"a": 1}, observation_id="obs_1")
        st.transform_observation({"a": 1}, observation_id="obs_2")

        assert mock_inner.transform.call_count == 2

    def test_cache_miss_when_id_is_none(self):
        st, mock_inner = _make_shared_transformer()
        st.transform_observation({"a": 1}, observation_id=None)
        st.transform_observation({"a": 1}, observation_id=None)

        assert mock_inner.transform.call_count == 2

    def test_clear_cache_resets_state(self):
        st, mock_inner = _make_shared_transformer()
        st.transform_observation({"a": 1}, observation_id="obs_1")
        st.clear_cache()

        assert st.transformed_observation is None
        assert st.last_observation_id is None

        st.transform_observation({"a": 1}, observation_id="obs_1")
        assert mock_inner.transform.call_count == 2


class TestLegacyNdarrayPath:
    def test_returns_ndarray_when_transform_returns_ndarray(self):
        st, mock_inner = _make_shared_transformer(arch="original")
        mock_inner.transform.return_value = np.zeros((4, 128, 128), dtype=np.float32)

        result = st.transform_observation({"a": 1}, observation_id="obs_0")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 128, 128)

    def test_legacy_path_does_not_call_transform_tactical(self):
        st, mock_inner = _make_shared_transformer(arch="original")
        mock_inner.transform.return_value = np.zeros((4, 128, 128), dtype=np.float32)

        st.transform_observation({"a": 1}, observation_id="obs_0")
        mock_inner.transform_tactical.assert_not_called()

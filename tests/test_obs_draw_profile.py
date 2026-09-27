"""Opt-in observation draw profiling (SC2_COLLECTOR_PROFILE / SC2_OBS_LAYER_PROFILE)."""

import numpy as np
import pytest

from src.transformers.observation_transformer import ObservationTransformer
from src.transformers.squad_observation_builder import SquadObservationBuilder
from src.utils.layer_constants import LayerConstants


def test_timed_phase_accumulates_totals(monkeypatch):
    monkeypatch.setenv("SC2_COLLECTOR_PROFILE", "1")
    tr = ObservationTransformer(
        map_size=(32, 32),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
        },
    )
    tr._obs_profile_flag = True
    with tr._timed_phase("g_creep"):
        pass
    with tr._timed_phase("u_dps"):
        _ = sum(range(1000))
    prof = tr.pop_obs_profile_means()
    assert "g_creep" in prof
    assert "u_dps" in prof


def test_builder_pop_merges_transformer_phases(monkeypatch):
    monkeypatch.setenv("SC2_COLLECTOR_PROFILE", "1")
    builder = SquadObservationBuilder(
        {
            "agent_architecture": "squad",
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "include_race": False,
        }
    )
    builder._obs_profile_flag = True
    builder._transformer._obs_profile_flag = True
    with builder._timed_phase("sb_mask"):
        pass
    with builder._transformer._timed_phase("g_path"):
        pass
    prof = builder.pop_obs_profile_means()
    assert "sb_mask" in prof
    assert "g_path" in prof


def test_creep_skipped_when_layer_inactive(monkeypatch):
    monkeypatch.setenv("SC2_COLLECTOR_PROFILE", "1")
    tr = ObservationTransformer(
        map_size=(32, 32),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
        },
    )
    tr.current_layer_map = LayerConstants.LAYERS
    tr.is_tactical_mode = False
    tr._update_active_indices()
    layers = np.zeros((LayerConstants.get_num_layers(), 8, 8), dtype=np.float32)
    creep_idx = LayerConstants.LAYERS[LayerConstants.TERRAIN_CREEP]
    # Force a fake creep payload; inactive layer must no-op without sampling.
    class _Creep:
        data = np.ones((16, 16), dtype=np.uint8)
        width = 16
        height = 16

    tr._draw_terrain_creep({"creep": _Creep(), "game_info": None}, layers)
    assert not layers[creep_idx].any()


def test_cache_stat_inc_flushed_in_pop(monkeypatch):
    monkeypatch.setenv("SC2_COLLECTOR_PROFILE", "1")
    tr = ObservationTransformer(
        map_size=(32, 32),
        training_conf={
            "agent_architecture": "universal",
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
        },
    )
    tr._obs_profile_flag = True
    tr._cache_stat_inc("tg_hit", 3)
    tr._cache_stat_inc("tg_miss", 1)
    prof = tr.pop_obs_profile_means()
    assert prof["cache_tg_hit"] == 3.0
    assert prof["cache_tg_miss"] == 1.0
    assert tr.pop_obs_profile_means() == {}


def test_builder_cache_stats_merge_with_transformer(monkeypatch):
    monkeypatch.setenv("SC2_COLLECTOR_PROFILE", "1")
    builder = SquadObservationBuilder(
        {
            "agent_architecture": "squad",
            "active_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "active_tactical_layers": [LayerConstants.ALLIED_UNIT_DENSITY],
            "include_race": False,
        }
    )
    builder._obs_profile_flag = True
    builder._transformer._obs_profile_flag = True
    builder._cache_stat_inc("gmap_swap", 2)
    builder._transformer._cache_stat_inc("tt_hit", 5)
    prof = builder.pop_obs_profile_means()
    assert prof["cache_gmap_swap"] == 2.0
    assert prof["cache_tt_hit"] == 5.0

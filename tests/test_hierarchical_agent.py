"""Tests for HierarchicalAgentSystem — agent creation and observation propagation."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


def _make_system(arch="universal", active_layers=None):
    """Create a HierarchicalAgentSystem with a mocked SharedObservationTransformer."""
    from src.hierarchical_agent import HierarchicalAgentSystem

    active_layers = active_layers or [
        "allied_unit_density", "enemy_unit_density",
        "terrain_pathable", "command_objective",
    ]
    training_conf = {
        "agent_architecture": arch,
        "active_layers": active_layers,
        "active_tactical_layers": active_layers,
    }

    with patch("src.hierarchical_agent.SharedObservationTransformer") as MockST:
        mock_st_instance = MockST.return_value
        mock_transformer = MagicMock()
        mock_transformer.active_layers = active_layers
        mock_transformer.active_tactical_layers = active_layers
        mock_st_instance.transformer = mock_transformer

        mock_st_instance.transform_observation.return_value = {
            "map": np.zeros((4, 128, 128), dtype=np.float32),
            "tactical": np.zeros((4, 32, 32), dtype=np.float32),
            "command": np.zeros(10, dtype=np.float32),
        }

        system = HierarchicalAgentSystem(
            map_size=(128, 128), grid_size=32, training_conf=training_conf,
        )
    return system, mock_st_instance


class TestCreateUnitAgent:
    def test_universal_creates_universal_agent(self):
        system, _ = _make_system(arch="universal")
        agent = system.create_unit_agent("agent_0")

        from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
        assert isinstance(agent, UniversalUnitControlAgent)

    def test_original_creates_base_agent(self):
        system, _ = _make_system(arch="original")
        agent = system.create_unit_agent("agent_0")

        from src.agent.unit_control_agent import UnitControlAgent
        from src.agent.universal_unit_control_agent import UniversalUnitControlAgent
        assert isinstance(agent, UnitControlAgent)
        assert not isinstance(agent, UniversalUnitControlAgent)

    def test_agent_is_stored_in_unit_agents(self):
        system, _ = _make_system()
        agent = system.create_unit_agent("my_agent")
        assert system.unit_agents["my_agent"] is agent

    def test_multiple_agents_independent(self):
        system, _ = _make_system()
        a1 = system.create_unit_agent("a1")
        a2 = system.create_unit_agent("a2")
        assert a1 is not a2
        assert len(system.unit_agents) == 2


class TestProcessObservation:
    def test_calls_shared_transformer_once(self):
        system, mock_st = _make_system()
        system.create_unit_agent("agent_0")

        raw_obs = {"game_info": MagicMock()}
        system.process_observation(raw_obs)

        mock_st.transform_observation.assert_called_once()

    def test_propagates_active_layers_to_agents(self):
        system, _ = _make_system(active_layers=["allied_unit_density", "enemy_unit_density"])
        agent = system.create_unit_agent("agent_0")

        raw_obs = {"game_info": MagicMock()}
        system.process_observation(raw_obs)

        assert agent.active_layers == ["allied_unit_density", "enemy_unit_density"]

    def test_propagates_active_tactical_layers_to_agents(self):
        system, _ = _make_system(active_layers=["terrain_pathable"])
        agent = system.create_unit_agent("agent_0")

        raw_obs = {"game_info": MagicMock()}
        system.process_observation(raw_obs)

        assert agent.active_tactical_layers == ["terrain_pathable"]

    def test_returns_results_for_all_agents(self):
        system, _ = _make_system()
        system.create_unit_agent("a1")
        system.create_unit_agent("a2")

        raw_obs = {"game_info": MagicMock()}
        results = system.process_observation(raw_obs)

        assert "a1" in results
        assert "a2" in results

    def test_observation_id_increments(self):
        system, mock_st = _make_system()
        system.create_unit_agent("agent_0")

        raw_obs = {"game_info": MagicMock()}
        system.process_observation(raw_obs)
        system.process_observation(raw_obs)

        calls = mock_st.transform_observation.call_args_list
        assert calls[0][1]["observation_id"] == "obs_0"
        assert calls[1][1]["observation_id"] == "obs_1"

    def test_passes_active_unit(self):
        system, mock_st = _make_system()
        system.create_unit_agent("agent_0")

        mock_unit = MagicMock()
        system.process_observation({"game_info": MagicMock()}, active_unit=mock_unit)

        _, kwargs = mock_st.transform_observation.call_args
        assert kwargs["active_unit"] is mock_unit

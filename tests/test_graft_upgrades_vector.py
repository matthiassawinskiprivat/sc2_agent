"""Trailing +6 upgrades graft on squad query_linear (after race)."""
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.agent.models.squad_network import SquadNetwork
from src.agent.squad_constants import MAX_SQUAD_UNITS, SQUAD_COMMAND_DIM, SQUAD_RACE_DIM, SQUAD_UPGRADES_DIM
from src.utils.model_utils import graft_model_weights


def _obs_space(*, race: bool, upgrades: bool, n_layers=3, n_tactical=3, feat=8):
    d = {
        "map": spaces.Box(0, 1, (n_layers, 128, 128), np.float32),
        "command": spaces.Box(0, 1, (SQUAD_COMMAND_DIM,), np.float32),
        "anchor_tactical": spaces.Box(0, 1, (n_tactical, 32, 32), np.float32),
        "unit_scalars": spaces.Box(0, 1, (MAX_SQUAD_UNITS, feat), np.float32),
        "unit_mask": spaces.Box(0, 1, (MAX_SQUAD_UNITS,), np.float32),
    }
    if race:
        d["race"] = spaces.Box(0, 1, (SQUAD_RACE_DIM,), np.float32)
    if upgrades:
        d["upgrades"] = spaces.Box(0, 1, (SQUAD_UPGRADES_DIM,), np.float32)
    return spaces.Dict(d)


def test_graft_query_linear_race_to_upgrades_trailing():
    old_net = SquadNetwork(_obs_space(race=True, upgrades=False), features_dim=64)
    new_net = SquadNetwork(_obs_space(race=True, upgrades=True), features_dim=64)
    old_w = old_net.query_linear[0].weight.data.clone()
    new_sd = new_net.state_dict()
    grafted = graft_model_weights(old_net.state_dict(), new_in_channels=3, target_state_dict=new_sd)
    # Find query_linear weight key
    keys = [k for k in grafted if "query_linear" in k and k.endswith("weight")]
    assert keys
    gw = grafted[keys[0]]
    assert gw.shape[1] == old_w.shape[1] + SQUAD_UPGRADES_DIM
    assert torch.allclose(gw[:, : old_w.shape[1]], old_w)

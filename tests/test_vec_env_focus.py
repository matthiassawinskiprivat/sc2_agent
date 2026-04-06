"""Obs focal unit tag for recurrent training (Level3/7 cycling)."""
import numpy as np
from unittest.mock import MagicMock

from train.env_utils import vec_env_obs_focus_tags
from train.envs.level_3_env import Level3Env


def test_vec_env_obs_focus_tags_from_env_method():
    vec = MagicMock()
    vec.env_method = MagicMock(return_value=[7, 8, 9])
    arr = vec_env_obs_focus_tags(vec, 3)
    assert arr.dtype == np.int64
    assert arr.tolist() == [7, 8, 9]


def test_level3_get_obs_focus_unit_tag_minimal():
    env = object.__new__(Level3Env)
    env.active_unit_tags = [100, 200, 300]
    env.current_unit_index = 2
    assert Level3Env.get_obs_focus_unit_tag(env) == 300


def test_level3_get_obs_focus_empty_tags():
    env = object.__new__(Level3Env)
    env.active_unit_tags = []
    env.current_unit_index = 0
    assert Level3Env.get_obs_focus_unit_tag(env) == -1

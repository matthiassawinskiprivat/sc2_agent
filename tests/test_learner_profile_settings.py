"""training.learner_profile parsing for MaskableRecurrentPPO.train() profiling."""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.mark.parametrize(
    "value,expect_enabled,expect_phases",
    [
        (None, True, False),
        (False, False, False),
        (True, True, False),
        ("false", False, False),
        ("0", False, False),
        ("off", False, False),
        ("no", False, False),
        ("true", True, False),
        ("1", True, False),
        ("on", True, False),
        ("yes", True, False),
        ("phases", True, True),
        ("2", True, True),
        ("detailed", True, True),
    ],
)
def test_parse_learner_profile_option(value, expect_enabled, expect_phases):
    from src.agent.maskable_recurrent_ppo import parse_learner_profile_option

    en, ph = parse_learner_profile_option(value)
    assert en is expect_enabled
    assert ph is expect_phases

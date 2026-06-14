"""Tests for SC2 window focus helper (no-op off Windows)."""

import sys

from src.utils.sc2_window_focus import try_focus_starcraft_window


def test_try_focus_starcraft_window_noop_off_windows():
    if sys.platform == "win32":
        # Cannot assert focus in headless CI; smoke-call only.
        try_focus_starcraft_window()
    else:
        assert try_focus_starcraft_window() is False

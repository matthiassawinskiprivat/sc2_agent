"""Best-effort focus for the StarCraft II client window (Windows debug UI launcher)."""

from __future__ import annotations

import sys


def try_focus_starcraft_window() -> bool:
    """
    Raise the SC2 game window on Windows so audio is not ducked behind the debug UI.

    Returns True if a matching window was found and focused.
    """
    if sys.platform != "win32":
        return False

    import ctypes

    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    ASFW_ANY = 0xFFFFFFFF
    found = False

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _enum_proc(hwnd, _lparam):
        nonlocal found
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.lower()
        if "starcraft" not in title:
            return True
        user32.AllowSetForegroundWindow(ASFW_ANY)
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        found = True
        return False

    user32.EnumWindows(_enum_proc, 0)
    return found

import time
import pytest
import threading
from unittest.mock import MagicMock
from src.utils.watchdog import Watchdog

def test_watchdog_start_rearms_if_already_running():
    event = threading.Event()

    def callback():
        event.set()

    watchdog = Watchdog(timeout=0.3, callback=callback)
    watchdog.start()
    watchdog.start()
    time.sleep(0.15)
    assert not event.is_set()
    triggered = event.wait(timeout=0.5)
    assert triggered, "Watchdog start() while running should rearm, not no-op"
    watchdog.stop()


def test_watchdog_triggers_on_timeout():
    """Test that watchdog triggers the callback after timeout."""
    event = threading.Event()
    def callback():
        event.set()
        
    timeout = 0.2
    watchdog = Watchdog(timeout=timeout, callback=callback)
    watchdog.start()
    
    # Wait for event with timeout
    triggered = event.wait(timeout=1.0)
    assert triggered, "Watchdog failed to trigger callback within expected time"
    watchdog.stop()

def test_watchdog_reset_prevents_trigger():
    """Test that resetting the watchdog prevents it from triggering."""
    event = threading.Event()
    def callback():
        event.set()
        
    timeout = 0.4
    watchdog = Watchdog(timeout=timeout, callback=callback)
    watchdog.start()
    
    # Wait half the timeout, then reset
    time.sleep(0.2)
    watchdog.reset()
    
    # Wait another 0.3s (total 0.5s from start, but only 0.3s from reset)
    # Callback should NOT have triggered yet (needs 0.4s from reset)
    triggered = event.wait(timeout=0.1)
    assert not triggered, "Watchdog triggered prematurely after reset"
    
    # Now wait for it to actually trigger
    triggered = event.wait(timeout=1.0)
    assert triggered, "Watchdog failed to trigger after reset"
    watchdog.stop()

def test_watchdog_stop_prevents_trigger():
    """Test that stopping the watchdog prevents it from triggering."""
    event = threading.Event()
    def callback():
        event.set()
        
    timeout = 0.2
    watchdog = Watchdog(timeout=timeout, callback=callback)
    watchdog.start()
    watchdog.stop()
    
    triggered = event.wait(timeout=0.5)
    assert not triggered, "Watchdog triggered after being stopped"

def test_watchdog_multiple_triggers():
    """Test that watchdog can trigger multiple times if configured."""
    count = 0
    event = threading.Event()
    def callback():
        nonlocal count
        count += 1
        if count >= 2:
            event.set()
            
    timeout = 0.2
    # If recurring=True, it should trigger again after first timeout
    watchdog = Watchdog(timeout=timeout, callback=callback, recurring=True)
    watchdog.start()
    
    # Wait for at least 2 triggers
    triggered = event.wait(timeout=1.5)
    assert triggered, f"Watchdog failed to trigger multiple times (count: {count})"
    assert count >= 2
    watchdog.stop()
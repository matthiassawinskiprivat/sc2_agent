import time
import pytest
from unittest.mock import MagicMock
from src.utils.watchdog import Watchdog

def test_watchdog_triggers_on_timeout():
    """Test that watchdog triggers the callback after timeout."""
    callback = MagicMock()
    timeout = 0.1
    
    watchdog = Watchdog(timeout=timeout, callback=callback)
    watchdog.start()
    
    # Wait for timeout + buffer
    time.sleep(0.2)
    
    callback.assert_called_once()
    watchdog.stop()

def test_watchdog_reset_prevents_trigger():
    """Test that resetting the watchdog prevents it from triggering."""
    callback = MagicMock()
    timeout = 0.2
    
    watchdog = Watchdog(timeout=timeout, callback=callback)
    watchdog.start()
    
    # Wait half the timeout, then reset
    time.sleep(0.1)
    watchdog.reset()
    
    # Wait another half timeout (total 0.2 since start, but 0.1 since reset)
    time.sleep(0.1)
    callback.assert_not_called()
    
    # Wait more to ensure it eventually triggers if not reset again
    time.sleep(0.15)
    callback.assert_called_once()
    watchdog.stop()

def test_watchdog_stop_prevents_trigger():
    """Test that stopping the watchdog prevents it from triggering."""
    callback = MagicMock()
    timeout = 0.1
    
    watchdog = Watchdog(timeout=timeout, callback=callback)
    watchdog.start()
    watchdog.stop()
    
    time.sleep(0.2)
    callback.assert_not_called()

def test_watchdog_multiple_triggers():
    """Test that watchdog can trigger multiple times if configured."""
    callback = MagicMock()
    timeout = 0.1
    
    # If recurring=True, it should trigger again after first timeout
    watchdog = Watchdog(timeout=timeout, callback=callback, recurring=True)
    watchdog.start()
    
    time.sleep(0.35) # Should trigger ~3 times
    assert callback.call_count >= 2
    watchdog.stop()

import threading
import time
import sys
import traceback
from typing import Callable, Optional
from loguru import logger

class Watchdog:
    """
    A simple watchdog timer that executes a callback if not reset within a timeout.
    Runs in a separate thread.
    """
    def __init__(self, timeout: float, callback: Optional[Callable] = None, recurring: bool = False):
        self.timeout = timeout
        self.callback = callback or self._default_callback
        self.recurring = recurring
        
        self._timer: Optional[threading.Timer] = None
        self._last_reset_time = 0.0
        self._is_running = False
        self._lock = threading.Lock()
        self._monitored_thread = threading.current_thread()

    def _default_callback(self):
        logger.warning(f"WATCHDOG TRIGGERED! Timeout of {self.timeout}s exceeded.")
        self.dump_stack_trace()

    def dump_stack_trace(self):
        """Logs the stack trace of the monitored thread."""
        logger.info(f"Dumping stack trace for thread: {self._monitored_thread.name}")
        # Note: sys._current_frames() returns a dict of thread_id -> stack_frame
        for thread_id, stack in sys._current_frames().items():
            if thread_id == self._monitored_thread.ident:
                extracted_stack = traceback.extract_stack(stack)
                formatted_stack = "".join(traceback.format_list(extracted_stack))
                logger.info(f"\n{formatted_stack}")
                break

    def _run_callback(self):
        with self._lock:
            if not self._is_running:
                return
            
        try:
            self.callback()
        except Exception as e:
            logger.error(f"Watchdog callback failed: {e}")

        if self.recurring:
            self.reset()

    def start(self):
        """Starts the watchdog timer."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._monitored_thread = threading.current_thread()
        self.reset()

    def stop(self):
        """Stops the watchdog timer."""
        with self._lock:
            self._is_running = False
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def reset(self):
        """Resets the watchdog timer."""
        with self._lock:
            if not self._is_running:
                return
            
            if self._timer:
                self._timer.cancel()
            
            self._timer = threading.Timer(self.timeout, self._run_callback)
            self._timer.daemon = True # Ensure it doesn't block process exit
            self._timer.start()
            self._last_reset_time = time.time()

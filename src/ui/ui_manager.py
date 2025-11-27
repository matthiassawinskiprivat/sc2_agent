"""
UI Manager for StarCraft 2 Unit Control Agent
Handles all UI-related functionality separate from core agent logic
Updated: Default resolution 512x512
"""

import sys
import os
import numpy as np
import time
import threading
import queue
import json
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.unit_control_agent import UnitControlAgent
from src.transformers.ui_transformer import RealTimeUITransformer
from src.utils.layer_constants import LayerConstants


class UIManager:
    """Manages UI functionality separate from core agent"""

    def __init__(self, agent: UnitControlAgent, display_size: tuple[int, int] = (512, 512)):
        """
        Initialize UI Manager

        Args:
            agent: Core unit control agent
            display_size: (width, height) for UI display. Default 512x512 (4x upscale for 128x128).
        """
        self.agent = agent
        self.display_size = display_size

        # UI components
        self.ui_transformer = RealTimeUITransformer(display_size=display_size)

        # Robustly load layers directly from the dictionary keys
        self.enabled_layers = list(LayerConstants.LAYERS.keys())

        # Threading
        self.update_queue = queue.Queue(maxsize=10)
        self.ui_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.ui_thread = None

        if hasattr(agent, 'ui_queue'):
            agent.ui_queue = self.update_queue
        else:
            setattr(agent, 'ui_queue', self.update_queue)

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self._observation_lock = threading.Lock()

    def start_real_time_mode(self):
        """Start real-time UI updates"""
        if self.is_running:
            return

        self.is_running = True
        self.ui_thread = threading.Thread(target=self._ui_update_loop)
        self.ui_thread.daemon = True
        self.ui_thread.start()
        print("Real-time UI mode started")

    def stop_real_time_mode(self):
        """Stop real-time UI updates"""
        if not self.is_running:
            return

        self.is_running = False
        if self.ui_thread:
            self.ui_thread.join(timeout=1.0)
        print("Real-time UI mode stopped")

    def _ui_update_loop(self):
        """Main UI update loop"""
        while self.is_running:
            try:
                # Get latest observation from agent
                with self._observation_lock:
                    current_obs = self.agent.transformed_observation

                    if current_obs is not None:
                        # Create UI visualization
                        # The UI Transformer internally handles the resizing from Obs Size -> display_size
                        # using Nearest Neighbor to keep it crisp.
                        frame = self.ui_transformer.update_display(
                            current_obs,
                            self.enabled_layers,
                            fps=self.current_fps
                        )

                        if frame is not None:
                            try:
                                self.ui_queue.put_nowait(frame)
                            except queue.Full:
                                try:
                                    self.ui_queue.get_nowait()
                                    self.ui_queue.put_nowait(frame)
                                except queue.Empty:
                                    pass

                # Update FPS
                self._update_fps()

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"UI update error: {e}")
                time.sleep(1.0)

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
        self.frame_count += 1

    def get_latest_ui_frame(self) -> Optional[np.ndarray]:
        """Get the latest UI frame for display"""
        try:
            return self.ui_queue.get_nowait()
        except queue.Empty:
            return None

    def set_layer_visibility(self, layer_name: str, visible: bool):
        """Set visibility of a specific layer"""
        if layer_name in self.enabled_layers:
            if not visible:
                self.enabled_layers.remove(layer_name)
        else:
            if visible and layer_name in LayerConstants.LAYERS:
                self.enabled_layers.append(layer_name)

    def toggle_layer_visibility(self, layer_name: str):
        """Toggle visibility of a specific layer"""
        if layer_name in self.enabled_layers:
            self.enabled_layers.remove(layer_name)
        else:
            self.enabled_layers.append(layer_name)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get information about all available layers"""
        layer_info = []

        for layer_name in LayerConstants.LAYERS:
            info = {
                'name': layer_name,
                'enabled': layer_name in self.enabled_layers,
                'color': self.ui_transformer.layer_colors.get(layer_name, [1.0, 1.0, 1.0]),
                'description': ""
            }
            layer_info.append(info)

        return layer_info

    def save_ui_debug_data(self, filename_prefix: str):
        """Save UI debug data"""
        if self.agent.transformed_observation is None:
            return

        try:
            os.makedirs("debug_output", exist_ok=True)

            # Save UI visualization
            frame = self.get_latest_ui_frame()
            if frame is not None:
                import cv2
                cv2.imwrite(f"debug_output/{filename_prefix}_ui.png", frame)

        except Exception as e:
            print(f"Error saving UI debug data: {e}")


class SimpleUIManager:
    """Simple UI manager for headless environments"""

    def __init__(self, agent: UnitControlAgent):
        self.agent = agent
        self.enabled_layers = list(LayerConstants.LAYERS.keys())
        self.is_running = False

    def start_real_time_mode(self):
        self.is_running = True
        print("Simple UI mode started (headless)")

    def stop_real_time_mode(self):
        self.is_running = False
        print("Simple UI mode stopped")

    def get_latest_ui_frame(self):
        return None

    def set_layer_visibility(self, layer_name: str, visible: bool):
        pass

    def get_layer_info(self):
        return [{'name': name, 'enabled': True} for name in LayerConstants.LAYERS]
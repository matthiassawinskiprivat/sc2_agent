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

    def __init__(
        self,
        agent: UnitControlAgent,
        global_display_size: tuple[int, int] = (768, 768),
        tactical_display_size: tuple[int, int] = (512, 512),
    ):
        """
        Initialize UI Manager

        Args:
            agent: Core unit control agent
            global_display_size: Upscaled size for the 128×128 global view.
            tactical_display_size: Upscaled size for the 32×32 tactical view.
        """
        self.agent = agent
        self.global_display_size = global_display_size
        self.tactical_display_size = tactical_display_size

        self.global_ui_transformer = RealTimeUITransformer(display_size=global_display_size)
        self.tactical_ui_transformer = RealTimeUITransformer(display_size=tactical_display_size)

        # Robustly load layers directly from the dictionary keys
        self.enabled_layers = list(LayerConstants.LAYERS.keys())
        self.enabled_tactical_layers = list(LayerConstants.TACTICAL_LAYERS.keys())

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
        self.active_layers = None

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
                    # Prefer explicitly prepared UI observation (e.g. global view), fallback to model input
                    global_obs = getattr(self.agent, "ui_observation", self.agent.transformed_observation)
                    tactical_obs = getattr(self.agent, "tactical_observation", None)

                    if global_obs is not None:
                        # Create UI visualization for global map
                        global_frame = self.global_ui_transformer.update_display(
                            global_obs,
                            self.enabled_layers,
                            fps=self.current_fps,
                            active_layers=getattr(self.agent, "active_layers", self.active_layers),
                        )
                        
                        # Create UI visualization for tactical map (if available)
                        tactical_frame = None
                        if tactical_obs is not None:
                            # Pruned tactical observations MUST have their layer names provided
                            # to correctly map indices back to UI colors.
                            tact_active = getattr(self.agent, "active_tactical_layers", None)
                            if tact_active is None:
                                # Default tactical view uses TACTICAL_LAYERS order
                                tact_active = sorted(LayerConstants.TACTICAL_LAYERS.keys(), 
                                                   key=lambda k: LayerConstants.TACTICAL_LAYERS[k])
                                                   
                            tactical_frame = self.tactical_ui_transformer.update_display(
                                tactical_obs,
                                self.enabled_tactical_layers,
                                active_layers=tact_active
                            )

                        if global_frame is not None:
                            combined_data = {
                                "global": global_frame,
                                "tactical": tactical_frame
                            }
                            try:
                                self.ui_queue.put_nowait(combined_data)
                            except queue.Full:
                                try:
                                    self.ui_queue.get_nowait()
                                    self.ui_queue.put_nowait(combined_data)
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

    def set_layer_visibility(self, layer_name: str, visible: bool, is_tactical: bool = False):
        """Set visibility of a specific layer"""
        target_list = self.enabled_tactical_layers if is_tactical else self.enabled_layers
        valid_map = LayerConstants.TACTICAL_LAYERS if is_tactical else LayerConstants.LAYERS

        if layer_name in target_list:
            if not visible:
                target_list.remove(layer_name)
        else:
            if visible and layer_name in valid_map:
                target_list.append(layer_name)

    def toggle_layer_visibility(self, layer_name: str, is_tactical: bool = False):
        """Toggle visibility of a specific layer"""
        target_list = self.enabled_tactical_layers if is_tactical else self.enabled_layers
        if layer_name in target_list:
            target_list.remove(layer_name)
        else:
            target_list.append(layer_name)

    def get_layer_info(self, is_tactical: bool = False) -> List[Dict[str, Any]]:
        """Get information about available layers"""
        layer_info = []
        target_map = LayerConstants.TACTICAL_LAYERS if is_tactical else LayerConstants.LAYERS
        enabled_list = self.enabled_tactical_layers if is_tactical else self.enabled_layers

        for layer_name in target_map:
            info = {
                'name': layer_name,
                'enabled': layer_name in enabled_list,
                'color': self.global_ui_transformer.layer_colors.get(
                    layer_name, [1.0, 1.0, 1.0]
                ),
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

    def set_layer_visibility(self, layer_name: str, visible: bool, is_tactical: bool = False):
        pass

    def get_layer_info(self, is_tactical: bool = False):
        target_map = LayerConstants.TACTICAL_LAYERS if is_tactical else LayerConstants.LAYERS
        return [{'name': name, 'enabled': True} for name in target_map]

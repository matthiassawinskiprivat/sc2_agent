import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from src.agent.unit_control_agent import UnitControlAgent
from src.utils.layer_constants import LayerConstants
from src.ui.ui_manager import UIManager


class DebugUI(QMainWindow):
    """Main debug UI window"""

    def __init__(self, agent: UnitControlAgent, ui_manager: UIManager = None):
        super().__init__()
        self.agent = agent
        self.ui_manager = ui_manager or UIManager(agent, display_size=(800, 600))

        # State
        self.is_realtime_mode = True
        self.current_frame = None

        self.init_ui()

        # Start UI manager
        self.ui_manager.start_real_time_mode()

        # Simple timer polling
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.poll_and_update)
        self.ui_timer.start(33)  # ~30 FPS

        # Statistics timer
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_statistics_display)
        self.stats_timer.start(1000)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("StarCraft 2 Unit Control Agent - Debug UI")
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel - Display area
        self.display_label = QLabel("Waiting for game data...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(800, 600)
        self.display_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.display_label.setScaledContents(True)

        # Right panel - Controls
        control_panel = self.create_control_panel()

        main_layout.addWidget(self.display_label, stretch=3)
        main_layout.addWidget(control_panel, stretch=1)

    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)

        # Layer controls
        layer_group = QGroupBox("Layer Controls")
        layer_layout = QVBoxLayout(layer_group)

        self.layer_checkboxes = {}
        for layer_name in LayerConstants.LAYERS:
            checkbox = QCheckBox(layer_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, name=layer_name:
                                          self.ui_manager.set_layer_visibility(name, state == Qt.Checked))
            layer_layout.addWidget(checkbox)
            self.layer_checkboxes[layer_name] = checkbox

        layout.addWidget(layer_group)

        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)

        # Real-time mode toggle
        self.realtime_button = QPushButton("Stop Real-time Mode")
        self.realtime_button.clicked.connect(self.toggle_realtime_mode)
        control_layout.addWidget(self.realtime_button)

        # Reset button
        reset_button = QPushButton("Reset Agent")
        reset_button.clicked.connect(self.agent.reset)
        control_layout.addWidget(reset_button)

        layout.addWidget(control_group)

        # Statistics
        stats_group = QGroupBox("Statistics")
        self.stats_layout = QVBoxLayout(stats_group)

        self.fps_label = QLabel("FPS: 0")
        self.frame_label = QLabel("Frames: 0")

        self.stats_layout.addWidget(self.fps_label)
        self.stats_layout.addWidget(self.frame_label)

        layout.addWidget(stats_group)
        layout.addStretch()

        return panel

    def poll_and_update(self):
        """Poll for frame and update display"""
        try:
            frame = self.ui_manager.get_latest_ui_frame()
            if frame is not None:
                self.update_display(frame)
        except Exception as e:
            print(f"DebugUI poll error: {e}")

    def update_display(self, frame: np.ndarray):
        """Update display with frame"""
        if frame is None:
            return

        try:
            # Ensure correct format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            # BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create QImage with COPY
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

            # Create pixmap
            pixmap = QPixmap.fromImage(q_image)

            # Update label
            self.display_label.setPixmap(pixmap)
            self.current_frame = frame

        except Exception as e:
            print(f"Display update error: {e}")
            import traceback
            traceback.print_exc()

    def update_statistics_display(self):
        """Update statistics"""
        try:
            stats = self.agent.get_observation_statistics()
            if stats:
                process_time = stats.get('processing_time_ms', 1.0)
                fps = 1000.0 / process_time if process_time > 0 else 0
                frames = stats.get('frame_count', 0)
                self.fps_label.setText(f"FPS: {fps:.1f}")
                self.frame_label.setText(f"Frames: {frames}")
        except:
            pass

    def toggle_realtime_mode(self):
        """Toggle real-time UI updates"""
        if self.is_realtime_mode:
            self.ui_manager.stop_real_time_mode()
            self.realtime_button.setText("Start Real-time Mode")
            self.is_realtime_mode = False
        else:
            self.ui_manager.start_real_time_mode()
            self.realtime_button.setText("Stop Real-time Mode")
            self.is_realtime_mode = True

    def closeEvent(self, event):
        """Clean shutdown"""
        self.is_realtime_mode = False
        self.ui_manager.stop_real_time_mode()
        event.accept()
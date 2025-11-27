import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, QGroupBox, QCheckBox)
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
        # Updated: Default to 512x512 for 4x upscale of 128x128 features
        self.ui_manager = ui_manager or UIManager(agent, display_size=(512, 512))

        # State
        self.is_realtime_mode = True
        self.current_frame = None

        # Track layer groups for bulk toggling
        self.allied_layer_names = []
        self.enemy_layer_names = []
        self.layer_checkboxes = {}

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
        # Adjusted window geometry
        self.setGeometry(100, 100, 1100, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel - Display area
        self.display_label = QLabel("Waiting for game data...")
        self.display_label.setAlignment(Qt.AlignCenter)

        # Enforce exact 512x512 size for the canvas
        self.display_label.setFixedSize(512, 512)

        self.display_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.display_label.setScaledContents(True)

        # Right panel - Controls
        control_panel = self.create_control_panel()

        main_layout.addWidget(self.display_label)
        main_layout.addWidget(control_panel, stretch=1)

    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)

        # Layer controls
        layer_group = QGroupBox("Layer Controls")

        # Grid Layout
        layer_layout = QGridLayout(layer_group)
        layer_layout.setVerticalSpacing(1)
        layer_layout.setHorizontalSpacing(10)
        # Ensure columns share width
        layer_layout.setColumnStretch(0, 1)
        layer_layout.setColumnStretch(1, 1)

        self.layer_checkboxes = {}

        # --- 1. Categorize Layers ---
        all_layers = list(LayerConstants.LAYERS.keys())
        self.allied_layer_names = []
        self.enemy_layer_names = []
        neutral_layers = []

        for name in all_layers:
            if name.startswith("allied") or name.startswith("command"):
                self.allied_layer_names.append(name)
            elif name.startswith("enemy"):
                self.enemy_layer_names.append(name)
            else:
                neutral_layers.append(name)

        # Helper to create styled checkbox
        def make_cb(layer_name):
            cb = QCheckBox(layer_name)
            cb.setChecked(True)
            cb.setStyleSheet("margin-bottom: 0px; margin-top: 0px; font-size: 10px;")
            cb.stateChanged.connect(lambda state, n=layer_name:
                                    self.ui_manager.set_layer_visibility(n, state == Qt.Checked))
            self.layer_checkboxes[layer_name] = cb
            return cb

        # --- 2. Add Master Checkboxes (Headers) ---

        # Master Allied
        master_allied = QCheckBox("Allied / Command (All)")
        master_allied.setChecked(True)
        master_allied.setStyleSheet("font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        master_allied.stateChanged.connect(self.toggle_all_allied_layers)
        layer_layout.addWidget(master_allied, 0, 0)

        # Master Enemy
        master_enemy = QCheckBox("Enemy Threat (All)")
        master_enemy.setChecked(True)
        master_enemy.setStyleSheet("font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        master_enemy.stateChanged.connect(self.toggle_all_enemy_layers)
        layer_layout.addWidget(master_enemy, 0, 1)

        # --- 3. Populate Columns Side-by-Side ---
        max_rows = max(len(self.allied_layer_names), len(self.enemy_layer_names))
        start_row = 1

        for i in range(max_rows):
            # Left Column (Allied)
            if i < len(self.allied_layer_names):
                cb = make_cb(self.allied_layer_names[i])
                layer_layout.addWidget(cb, start_row + i, 0)

            # Right Column (Enemy)
            if i < len(self.enemy_layer_names):
                cb = make_cb(self.enemy_layer_names[i])
                layer_layout.addWidget(cb, start_row + i, 1)

        # --- 4. Populate Neutral/Environment Below ---
        env_row_start = start_row + max_rows + 1
        lbl_env = QLabel("<b>Environment / Neutral</b>")
        lbl_env.setStyleSheet("margin-top: 10px;")
        # Span header across 2 columns
        layer_layout.addWidget(lbl_env, env_row_start, 0, 1, 2)

        env_content_start = env_row_start + 1
        for i, name in enumerate(neutral_layers):
            cb = make_cb(name)
            # Fill left then right
            row = env_content_start + (i // 2)
            col = i % 2
            layer_layout.addWidget(cb, row, col)

        layout.addWidget(layer_group)

        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)

        self.realtime_button = QPushButton("Stop Real-time Mode")
        self.realtime_button.clicked.connect(self.toggle_realtime_mode)
        control_layout.addWidget(self.realtime_button)

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

    def toggle_all_allied_layers(self, state):
        """Enable/Disable all allied layer checkboxes"""
        is_checked = (state == Qt.Checked)
        for name in self.allied_layer_names:
            if name in self.layer_checkboxes:
                # This programmatically triggers stateChanged, updating the UIManager
                self.layer_checkboxes[name].setChecked(is_checked)

    def toggle_all_enemy_layers(self, state):
        """Enable/Disable all enemy layer checkboxes"""
        is_checked = (state == Qt.Checked)
        for name in self.enemy_layer_names:
            if name in self.layer_checkboxes:
                self.layer_checkboxes[name].setChecked(is_checked)

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

    def update_statistics_display(self):
        """Update statistics"""
        try:
            stats = self.agent.get_observation_statistics() if hasattr(self.agent, "get_observation_statistics") else {}
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
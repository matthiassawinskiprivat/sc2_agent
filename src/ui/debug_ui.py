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
        # Default to 512x512 for 4x upscale of 128x128 features
        self.ui_manager = ui_manager or UIManager(agent, display_size=(512, 512))

        # State
        self.is_realtime_mode = True
        self.current_frame = None
        self.show_mask = False
        self.show_clicks = True

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
        self.setGeometry(100, 100, 1100, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel - Display area
        self.display_label = QLabel("Waiting for game data...")
        self.display_label.setAlignment(Qt.AlignCenter)
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
        layer_layout.setColumnStretch(0, 1)
        layer_layout.setColumnStretch(1, 1)

        self.layer_checkboxes = {}

        all_layers = list(LayerConstants.LAYERS.keys())
        self.allied_layer_names = [n for n in all_layers if n.startswith("allied") or n.startswith("command")]
        self.enemy_layer_names = [n for n in all_layers if n.startswith("enemy")]
        neutral_layers = [n for n in all_layers if n not in self.allied_layer_names and n not in self.enemy_layer_names]

        def make_cb(layer_name):
            cb = QCheckBox(layer_name)
            cb.setChecked(True)
            cb.setStyleSheet("margin-bottom: 0px; margin-top: 0px; font-size: 10px;")
            cb.stateChanged.connect(lambda state, n=layer_name:
                                    self.ui_manager.set_layer_visibility(n, state == Qt.Checked))
            self.layer_checkboxes[layer_name] = cb
            return cb

        # Headers & Master Checkboxes
        master_allied = QCheckBox("Allied / Command (All)")
        master_allied.setChecked(True)
        master_allied.setStyleSheet("font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        master_allied.stateChanged.connect(self.toggle_all_allied_layers)
        layer_layout.addWidget(master_allied, 0, 0)

        master_enemy = QCheckBox("Enemy Threat (All)")
        master_enemy.setChecked(True)
        master_enemy.setStyleSheet("font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        master_enemy.stateChanged.connect(self.toggle_all_enemy_layers)
        layer_layout.addWidget(master_enemy, 0, 1)

        # Columns
        max_rows = max(len(self.allied_layer_names), len(self.enemy_layer_names))
        start_row = 1

        for i in range(max_rows):
            if i < len(self.allied_layer_names):
                layer_layout.addWidget(make_cb(self.allied_layer_names[i]), start_row + i, 0)
            if i < len(self.enemy_layer_names):
                layer_layout.addWidget(make_cb(self.enemy_layer_names[i]), start_row + i, 1)

        # Neutral
        env_row_start = start_row + max_rows + 1
        lbl_env = QLabel("<b>Environment / Neutral</b>")
        lbl_env.setStyleSheet("margin-top: 10px;")
        layer_layout.addWidget(lbl_env, env_row_start, 0, 1, 2)

        env_content_start = env_row_start + 1
        for i, name in enumerate(neutral_layers):
            row = env_content_start + (i // 2)
            col = i % 2
            layer_layout.addWidget(make_cb(name), row, col)

        layout.addWidget(layer_group)

        # Overlay Controls
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout(overlay_group)

        # Mask Overlay
        self.mask_checkbox = QCheckBox("Show Action Mask (Red = Invalid)")
        self.mask_checkbox.setChecked(False)
        self.mask_checkbox.setStyleSheet("color: red; font-weight: bold;")
        self.mask_checkbox.stateChanged.connect(lambda s: setattr(self, 'show_mask', s == Qt.Checked))
        overlay_layout.addWidget(self.mask_checkbox)

        # Click Overlay
        self.click_checkbox = QCheckBox("Show Agent Clicks (Green Crosshair)")
        self.click_checkbox.setChecked(True)
        self.click_checkbox.setStyleSheet("color: green; font-weight: bold;")
        self.click_checkbox.stateChanged.connect(lambda s: setattr(self, 'show_clicks', s == Qt.Checked))
        overlay_layout.addWidget(self.click_checkbox)

        layout.addWidget(overlay_group)

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
        is_checked = (state == Qt.Checked)
        for name in self.allied_layer_names:
            if name in self.layer_checkboxes:
                self.layer_checkboxes[name].setChecked(is_checked)

    def toggle_all_enemy_layers(self, state):
        is_checked = (state == Qt.Checked)
        for name in self.enemy_layer_names:
            if name in self.layer_checkboxes:
                self.layer_checkboxes[name].setChecked(is_checked)

    def poll_and_update(self):
        try:
            frame = self.ui_manager.get_latest_ui_frame()
            if frame is not None:
                self.update_display(frame)
        except Exception as e:
            print(f"DebugUI poll error: {e}")

    def update_display(self, frame: np.ndarray):
        if frame is None: return

        try:
            # frame is 512x512 BGR (from UI Transformer)
            display_h, display_w, _ = frame.shape

            # 1. Overlay Action Mask
            if self.show_mask:
                mask_flat = self.agent.get_action_mask()
                if mask_flat is not None and mask_flat.size == (128 * 128):
                    mask_2d = mask_flat.reshape((128, 128))
                    # Upscale to 512x512 Nearest Neighbor
                    mask_resized = cv2.resize(mask_2d.astype(np.uint8), (display_w, display_h),
                                              interpolation=cv2.INTER_NEAREST)

                    # Tint Invalid Areas Red
                    invalid_area = (mask_resized == 0)
                    # Mix existing color with Red: (0.7 * Original) + (0.3 * Red)
                    # Or simpler: Max Red channel, zero others for strong indicator
                    # Let's do a semi-transparent red overlay

                    overlay = frame.copy()
                    overlay[invalid_area] = [0, 0, 200]  # BGR Red
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # 2. Overlay Last Click
            if self.show_clicks and self.agent.last_action is not None:
                idx = self.agent.last_action
                
                # Dynamic grid size (e.g. 32 or 128)
                grid_size = getattr(self.agent, 'grid_size', 128)
                
                # Convert 1D index to 2D
                y_grid, x_grid = divmod(idx, grid_size)

                # Scale up to 512x512
                scale = display_w / float(grid_size)
                cx = int(x_grid * scale + scale / 2)
                cy = int(y_grid * scale + scale / 2)

                # Draw Crosshair (Green)
                color = (0, 255, 0)  # BGR
                cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 2)
                cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 2)
                # Small white dot in exact center
                cv2.circle(frame, (cx, cy), 1, (255, 255, 255), -1)

            # Convert to QImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_frame.data, display_w, display_h, 3 * display_w, QImage.Format_RGB888).copy()

            self.display_label.setPixmap(QPixmap.fromImage(q_image))
            self.current_frame = frame

        except Exception as e:
            print(f"Display update error: {e}")

    def update_statistics_display(self):
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
        if self.is_realtime_mode:
            self.ui_manager.stop_real_time_mode()
            self.realtime_button.setText("Start Real-time Mode")
            self.is_realtime_mode = False
        else:
            self.ui_manager.start_real_time_mode()
            self.realtime_button.setText("Stop Real-time Mode")
            self.is_realtime_mode = True

    def closeEvent(self, event):
        self.is_realtime_mode = False
        self.ui_manager.stop_real_time_mode()
        event.accept()
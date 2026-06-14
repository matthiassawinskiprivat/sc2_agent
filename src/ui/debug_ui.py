import sys
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, QGroupBox, QCheckBox,
                             QComboBox, QStackedWidget, QLineEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIntValidator
from sc2.data import Race

from src.agent.unit_control_agent import UnitControlAgent
from src.utils.layer_constants import LayerConstants
from src.ui.ui_manager import UIManager
from src.ui.spatial_debug_control import parse_grid_field, SPATIAL_GRID_SIZE

GLOBAL_DISPLAY_SIZE = (768, 768)
TACTICAL_DISPLAY_SIZE = (512, 512)
RACE_UI_OPTIONS: Tuple[str, ...] = ("Terran", "Zerg", "Protoss", "Random")
DEFAULT_ALLIED_RACE = "Terran"
DEFAULT_ENEMY_RACE = "Zerg"
MODE_FREE_PLAY = "free_play"
MODE_TRAINING_LEVEL = "training_level"


def race_from_ui_label(label: str) -> Race:
    """Map debug UI race labels to ``sc2.data.Race`` values."""
    mapping = {
        "Terran": Race.Terran,
        "Zerg": Race.Zerg,
        "Protoss": Race.Protoss,
        "Random": Race.Random,
    }
    return mapping[label]


def discover_maps(maps_dir: str) -> List[str]:
    """List map stems from a directory of *.SC2Map files."""
    path = Path(maps_dir)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / maps_dir
    if not path.exists():
        return []
    return sorted(f.stem for f in path.glob("*.SC2Map"))


class DebugUI(QMainWindow):
    """Main debug UI window"""

    def __init__(
        self,
        agent: UnitControlAgent,
        ui_manager: UIManager = None,
        maps_dir: Optional[str] = None,
        on_start_game: Optional[Callable[[str, Race, Race], None]] = None,
        on_start_training_level: Optional[Callable[[str, str], None]] = None,
        pipeline_levels: Optional[List[str]] = None,
        on_training_level_changed: Optional[Callable[[str], List[str]]] = None,
        on_manual_spatial_move: Optional[Callable[[str, int, int], None]] = None,
        default_allied_race: str = DEFAULT_ALLIED_RACE,
        default_enemy_race: str = DEFAULT_ENEMY_RACE,
        default_map_name: Optional[str] = None,
    ):
        super().__init__()
        self.agent = agent
        self.ui_manager = ui_manager or UIManager(
            agent,
            global_display_size=GLOBAL_DISPLAY_SIZE,
            tactical_display_size=TACTICAL_DISPLAY_SIZE,
        )
        self.maps_dir = maps_dir
        self.on_start_game = on_start_game
        self.on_start_training_level = on_start_training_level
        self.pipeline_levels = list(pipeline_levels or [])
        self.on_training_level_changed = on_training_level_changed
        self.on_manual_spatial_move = on_manual_spatial_move
        self.default_allied_race = default_allied_race
        self.default_enemy_race = default_enemy_race
        self.default_map_name = default_map_name
        self.game_mode = MODE_FREE_PLAY
        self.mode_selector = None
        self.start_controls_stack = None
        self.map_selector = None
        self.allied_race_selector = None
        self.enemy_race_selector = None
        self.level_selector = None
        self.training_map_selector = None
        self.start_game_button = None
        self.spatial_control_group = None
        self.spatial_x_input = None
        self.spatial_y_input = None
        self.spatial_move_global_btn = None
        self.spatial_move_local_btn = None

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

    @staticmethod
    def _partition_layer_names(layer_names: List[str]):
        allied = [n for n in layer_names if n.startswith("allied") or n.startswith("command")]
        enemy = [n for n in layer_names if n.startswith("enemy")]
        neutral = [
            n for n in layer_names
            if n not in allied and n not in enemy
        ]
        return allied, enemy, neutral

    def _build_layer_checkbox_grid(
        self,
        layer_names: List[str],
        checkbox_store: Dict[str, QCheckBox],
        is_tactical: bool,
    ) -> QGridLayout:
        """Allied / enemy columns with master toggles, same layout as global view."""
        allied_names, enemy_names, neutral_names = self._partition_layer_names(layer_names)
        grid = QGridLayout()
        grid.setVerticalSpacing(1)
        grid.setHorizontalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        def make_cb(layer_name: str) -> QCheckBox:
            cb = QCheckBox(layer_name)
            cb.setChecked(True)
            cb.setStyleSheet("margin-bottom: 0px; margin-top: 0px; font-size: 10px;")
            cb.stateChanged.connect(
                lambda state, n=layer_name: self.ui_manager.set_layer_visibility(
                    n, state == Qt.Checked, is_tactical=is_tactical
                )
            )
            checkbox_store[layer_name] = cb
            return cb

        master_allied = QCheckBox("Allied / Command (All)")
        master_allied.setChecked(True)
        master_allied.setStyleSheet("font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        if is_tactical:
            master_allied.stateChanged.connect(self.toggle_all_allied_tactical_layers)
        else:
            master_allied.stateChanged.connect(self.toggle_all_allied_layers)
        grid.addWidget(master_allied, 0, 0)

        master_enemy = QCheckBox("Enemy Threat (All)")
        master_enemy.setChecked(True)
        master_enemy.setStyleSheet("font-weight: bold; font-size: 11px; margin-bottom: 5px;")
        if is_tactical:
            master_enemy.stateChanged.connect(self.toggle_all_enemy_tactical_layers)
        else:
            master_enemy.stateChanged.connect(self.toggle_all_enemy_layers)
        grid.addWidget(master_enemy, 0, 1)

        max_rows = max(len(allied_names), len(enemy_names))
        start_row = 1
        for i in range(max_rows):
            if i < len(allied_names):
                grid.addWidget(make_cb(allied_names[i]), start_row + i, 0)
            if i < len(enemy_names):
                grid.addWidget(make_cb(enemy_names[i]), start_row + i, 1)

        if neutral_names:
            env_row_start = start_row + max_rows + 1
            lbl_env = QLabel("<b>Environment / Neutral</b>")
            lbl_env.setStyleSheet("margin-top: 10px;")
            grid.addWidget(lbl_env, env_row_start, 0, 1, 2)
            env_content_start = env_row_start + 1
            for i, name in enumerate(neutral_names):
                row = env_content_start + (i // 2)
                col = i % 2
                grid.addWidget(make_cb(name), row, col)

        if is_tactical:
            self.tactical_allied_layer_names = allied_names
            self.tactical_enemy_layer_names = enemy_names
        else:
            self.allied_layer_names = allied_names
            self.enemy_layer_names = enemy_names

        return grid

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("StarCraft 2 Unit Control Agent - Debug UI")
        self.setGeometry(100, 100, 1820, 820)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel — global and tactical maps side by side (avoids vertical overlap)
        display_panel = QWidget()
        display_layout = QHBoxLayout(display_panel)
        display_layout.setSpacing(12)

        self.display_label = QLabel("Waiting for global data...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setFixedSize(*GLOBAL_DISPLAY_SIZE)
        self.display_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.display_label.setScaledContents(True)

        self.tactical_label = QLabel("Waiting for tactical data...")
        self.tactical_label.setAlignment(Qt.AlignCenter)
        self.tactical_label.setFixedSize(*TACTICAL_DISPLAY_SIZE)
        self.tactical_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.tactical_label.setScaledContents(True)

        global_column = QWidget()
        global_column_layout = QVBoxLayout(global_column)
        global_column_layout.setContentsMargins(0, 0, 0, 0)
        global_column_layout.addWidget(QLabel("<b>Global View (128x128)</b>"))
        global_column_layout.addWidget(self.display_label)
        global_column_layout.addStretch()

        tactical_column = QWidget()
        tactical_column_layout = QVBoxLayout(tactical_column)
        tactical_column_layout.setContentsMargins(0, 0, 0, 0)
        tactical_column_layout.addWidget(QLabel("<b>Tactical View (32x32)</b>"))
        tactical_column_layout.addWidget(self.tactical_label)
        tactical_column_layout.addWidget(self._create_spatial_control_group())
        tactical_column_layout.addStretch()

        display_layout.addWidget(global_column)
        display_layout.addWidget(tactical_column)

        # Right panel - Controls
        control_panel = self.create_control_panel()

        main_layout.addWidget(display_panel)
        main_layout.addWidget(control_panel, stretch=1)

    def _create_spatial_control_group(self) -> QGroupBox:
        """Manual spatial-index moves using the same decode path as the universal policy."""
        group = QGroupBox("Spatial test (grid 0–31)")
        layout = QVBoxLayout(group)

        hint = QLabel(
            "Cell (x, y) = spatial decoder index (gx, gy). "
            "move_global (0, 0) → map NW corner; (31, 31) → SE corner."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #555;")
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.addWidget(QLabel("x:"))
        self.spatial_x_input = QLineEdit("0")
        self.spatial_x_input.setFixedWidth(48)
        self.spatial_x_input.setValidator(QIntValidator(0, SPATIAL_GRID_SIZE - 1, self))
        row.addWidget(self.spatial_x_input)
        row.addWidget(QLabel("y:"))
        self.spatial_y_input = QLineEdit("0")
        self.spatial_y_input.setFixedWidth(48)
        self.spatial_y_input.setValidator(QIntValidator(0, SPATIAL_GRID_SIZE - 1, self))
        row.addWidget(self.spatial_y_input)
        row.addStretch()
        layout.addLayout(row)

        btn_row = QHBoxLayout()
        self.spatial_move_global_btn = QPushButton("move_global")
        self.spatial_move_global_btn.clicked.connect(
            lambda: self._emit_manual_spatial_move("move_global")
        )
        btn_row.addWidget(self.spatial_move_global_btn)
        self.spatial_move_local_btn = QPushButton("move_local")
        self.spatial_move_local_btn.clicked.connect(
            lambda: self._emit_manual_spatial_move("move_local")
        )
        btn_row.addWidget(self.spatial_move_local_btn)
        layout.addLayout(btn_row)

        group.setEnabled(False)
        self.spatial_control_group = group
        return group

    def _read_spatial_grid_coords(self) -> Tuple[int, int]:
        gx = parse_grid_field(self.spatial_x_input.text() if self.spatial_x_input else "0")
        gy = parse_grid_field(self.spatial_y_input.text() if self.spatial_y_input else "0")
        return gx, gy

    def _emit_manual_spatial_move(self, command_name: str) -> None:
        if not self.on_manual_spatial_move:
            return
        gx, gy = self._read_spatial_grid_coords()
        self.on_manual_spatial_move(command_name, gx, gy)

    def create_control_panel(self):
        """Create the control panel"""
        panel = QWidget()
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)

        if self.maps_dir and (self.on_start_game or self.on_start_training_level):
            game_group = QGroupBox("Game")
            game_layout = QVBoxLayout(game_group)

            mode_row = QHBoxLayout()
            mode_row.addWidget(QLabel("Mode:"))
            self.mode_selector = QComboBox()
            self.mode_selector.addItem("Free Play (vs Computer)", MODE_FREE_PLAY)
            self.mode_selector.addItem("Training Level Preview", MODE_TRAINING_LEVEL)
            self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
            mode_row.addWidget(self.mode_selector, stretch=1)
            game_layout.addLayout(mode_row)

            self.start_controls_stack = QStackedWidget()

            free_play_page = QWidget()
            free_play_layout = QVBoxLayout(free_play_page)
            free_play_layout.setContentsMargins(0, 0, 0, 0)

            free_play_layout.addWidget(QLabel("Map:"))
            self.map_selector = QComboBox()
            available_maps = discover_maps(self.maps_dir)
            self.map_selector.addItems(available_maps)
            if not available_maps:
                self.map_selector.addItem("(no maps found)")
            elif self.default_map_name:
                default_idx = self.map_selector.findText(self.default_map_name)
                if default_idx >= 0:
                    self.map_selector.setCurrentIndex(default_idx)
            free_play_layout.addWidget(self.map_selector)

            race_row = QHBoxLayout()
            allied_label = QLabel("Allied race:")
            self.allied_race_selector = QComboBox()
            self.allied_race_selector.addItems(list(RACE_UI_OPTIONS))
            allied_idx = RACE_UI_OPTIONS.index(self.default_allied_race) if self.default_allied_race in RACE_UI_OPTIONS else 0
            self.allied_race_selector.setCurrentIndex(allied_idx)
            race_row.addWidget(allied_label)
            race_row.addWidget(self.allied_race_selector, stretch=1)
            free_play_layout.addLayout(race_row)

            enemy_row = QHBoxLayout()
            enemy_label = QLabel("Enemy race:")
            self.enemy_race_selector = QComboBox()
            self.enemy_race_selector.addItems(list(RACE_UI_OPTIONS))
            enemy_idx = RACE_UI_OPTIONS.index(self.default_enemy_race) if self.default_enemy_race in RACE_UI_OPTIONS else 0
            self.enemy_race_selector.setCurrentIndex(enemy_idx)
            enemy_row.addWidget(enemy_label)
            enemy_row.addWidget(self.enemy_race_selector, stretch=1)
            free_play_layout.addLayout(enemy_row)

            self.start_controls_stack.addWidget(free_play_page)

            training_page = QWidget()
            training_layout = QVBoxLayout(training_page)
            training_layout.setContentsMargins(0, 0, 0, 0)

            training_layout.addWidget(QLabel("Level:"))
            self.level_selector = QComboBox()
            if self.pipeline_levels:
                self.level_selector.addItems(self.pipeline_levels)
            else:
                self.level_selector.addItem("(no pipeline levels)")
            self.level_selector.currentIndexChanged.connect(self._on_training_level_changed)
            training_layout.addWidget(self.level_selector)

            training_layout.addWidget(QLabel("Map:"))
            self.training_map_selector = QComboBox()
            training_layout.addWidget(self.training_map_selector)
            self._refresh_training_maps()

            self.start_controls_stack.addWidget(training_page)
            game_layout.addWidget(self.start_controls_stack)

            self.start_game_button = QPushButton("Start Game")
            self.start_game_button.clicked.connect(self._on_start_game_clicked)
            self.start_game_button.setEnabled(self._start_enabled_for_current_mode())
            game_layout.addWidget(self.start_game_button)

            layout.addWidget(game_group)

        # Layer controls — global
        layer_group = QGroupBox("Global View Layers")
        all_layers = list(LayerConstants.LAYERS.keys())
        self.layer_checkboxes = {}
        layer_group.setLayout(
            self._build_layer_checkbox_grid(all_layers, self.layer_checkboxes, is_tactical=False)
        )
        layout.addWidget(layer_group)

        # Layer controls — tactical (same allied/enemy layout as global)
        tactical_group = QGroupBox("Tactical View Layers")
        tactical_layers = list(LayerConstants.TACTICAL_LAYERS.keys())
        self.tactical_layer_checkboxes = {}
        tactical_group.setLayout(
            self._build_layer_checkbox_grid(
                tactical_layers, self.tactical_layer_checkboxes, is_tactical=True
            )
        )
        layout.addWidget(tactical_group)

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
        self.click_checkbox = QCheckBox("Show Click Targets (Green Crosshair)")
        self.click_checkbox.setChecked(True)
        self.click_checkbox.setStyleSheet("color: green; font-weight: bold;")
        self.click_checkbox.stateChanged.connect(lambda s: setattr(self, 'show_clicks', s == Qt.Checked))
        overlay_layout.addWidget(self.click_checkbox)

        layout.addWidget(overlay_group)

        layout.addStretch()
        return panel

    def get_allied_race(self) -> Race:
        if self.allied_race_selector is None:
            return race_from_ui_label(self.default_allied_race)
        return race_from_ui_label(self.allied_race_selector.currentText())

    def get_enemy_race(self) -> Race:
        if self.enemy_race_selector is None:
            return race_from_ui_label(self.default_enemy_race)
        return race_from_ui_label(self.enemy_race_selector.currentText())

    def get_selected_map_name(self) -> Optional[str]:
        if self.game_mode == MODE_TRAINING_LEVEL:
            if self.training_map_selector is None:
                return None
            map_name = self.training_map_selector.currentText()
        else:
            if self.map_selector is None:
                return None
            map_name = self.map_selector.currentText()
        if not map_name or map_name.startswith("(no "):
            return None
        return map_name

    def get_selected_level_name(self) -> Optional[str]:
        if self.level_selector is None:
            return None
        level_name = self.level_selector.currentText()
        if not level_name or level_name.startswith("(no "):
            return None
        return level_name

    def _current_mode(self) -> str:
        if self.mode_selector is None:
            return MODE_FREE_PLAY
        return self.mode_selector.currentData() or MODE_FREE_PLAY

    def _on_mode_changed(self, _index: int = 0) -> None:
        self.game_mode = self._current_mode()
        if self.start_controls_stack is not None:
            self.start_controls_stack.setCurrentIndex(
                1 if self.game_mode == MODE_TRAINING_LEVEL else 0
            )
        if self.start_game_button is not None:
            self.start_game_button.setEnabled(self._start_enabled_for_current_mode())

    def _on_training_level_changed(self, _index: int = 0) -> None:
        self._refresh_training_maps()
        if self.start_game_button is not None:
            self.start_game_button.setEnabled(self._start_enabled_for_current_mode())

    def _refresh_training_maps(self) -> None:
        if self.training_map_selector is None:
            return
        level_name = self.get_selected_level_name()
        maps: List[str] = []
        if level_name and self.on_training_level_changed:
            maps = list(self.on_training_level_changed(level_name))
        self.training_map_selector.blockSignals(True)
        self.training_map_selector.clear()
        if maps:
            self.training_map_selector.addItems(maps)
        else:
            self.training_map_selector.addItem("(no maps found)")
        self.training_map_selector.blockSignals(False)

    def _start_enabled_for_current_mode(self) -> bool:
        if self.game_mode == MODE_TRAINING_LEVEL:
            return (
                self.get_selected_level_name() is not None
                and self.get_selected_map_name() is not None
            )
        return discover_maps(self.maps_dir or "") != [] if self.maps_dir else False

    def _on_start_game_clicked(self):
        if self.game_mode == MODE_TRAINING_LEVEL:
            if not self.on_start_training_level:
                return
            level_name = self.get_selected_level_name()
            map_name = self.get_selected_map_name()
            if not level_name or not map_name:
                return
            self.on_start_training_level(level_name, map_name)
            return
        if not self.on_start_game or self.map_selector is None:
            return
        map_name = self.get_selected_map_name()
        if not map_name:
            return
        self.on_start_game(map_name, self.get_allied_race(), self.get_enemy_race())

    def set_game_controls_enabled(self, enabled: bool):
        if self.mode_selector is not None:
            self.mode_selector.setEnabled(enabled)
        if self.map_selector is not None:
            self.map_selector.setEnabled(enabled)
        if self.allied_race_selector is not None:
            self.allied_race_selector.setEnabled(enabled)
        if self.enemy_race_selector is not None:
            self.enemy_race_selector.setEnabled(enabled)
        if self.level_selector is not None:
            self.level_selector.setEnabled(enabled)
        if self.training_map_selector is not None:
            self.training_map_selector.setEnabled(enabled)
        if self.start_game_button is not None:
            self.start_game_button.setEnabled(
                enabled and self._start_enabled_for_current_mode()
            )
            self.start_game_button.setText("Start Game" if enabled else "Game running...")
        if self.spatial_control_group is not None:
            self.spatial_control_group.setEnabled(not enabled)

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

    def toggle_all_allied_tactical_layers(self, state):
        is_checked = (state == Qt.Checked)
        for name in self.tactical_allied_layer_names:
            if name in self.tactical_layer_checkboxes:
                self.tactical_layer_checkboxes[name].setChecked(is_checked)

    def toggle_all_enemy_tactical_layers(self, state):
        is_checked = (state == Qt.Checked)
        for name in self.tactical_enemy_layer_names:
            if name in self.tactical_layer_checkboxes:
                self.tactical_layer_checkboxes[name].setChecked(is_checked)

    def poll_and_update(self):
        try:
            frame = self.ui_manager.get_latest_ui_frame()
            if frame is not None:
                self.update_display(frame)
        except Exception as e:
            print(f"DebugUI poll error: {e}")

    def update_display(self, data: Dict[str, np.ndarray]):
        if data is None: return

        try:
            global_frame = data.get("global")
            tactical_frame = data.get("tactical")

            if global_frame is not None:
                self._render_to_label(global_frame, self.display_label, is_tactical=False)
            
            if tactical_frame is not None:
                self._render_to_label(tactical_frame, self.tactical_label, is_tactical=True)
            else:
                self.tactical_label.setText("Tactical view not active (No unit selected)")

        except Exception as e:
            print(f"Display update error: {e}")

    def _draw_click_overlay(self, rendered_frame: np.ndarray, label: QLabel, is_tactical: bool) -> None:
        click = getattr(self.agent, "last_debug_click", None)
        if click is None:
            return

        # Scale from obs pixels to the rendered frame, not the QLabel widget.
        display_h, display_w = rendered_frame.shape[:2]
        obs_size = 32 if is_tactical else 128

        if hasattr(self.agent, "world_to_cnn_pixel") and "global_cnn" in click and "tactical_cnn" in click:
            if is_tactical and hasattr(self.agent, "world_to_tactical_cnn_pixel"):
                world = click.get("world")
                active = getattr(self.agent, "debug_active_unit", None)
                if world is not None and active is not None:
                    px, py = self.agent.world_to_tactical_cnn_pixel(
                        float(world[0]), float(world[1]), active
                    )
                else:
                    px, py = click["tactical_cnn"]
            else:
                px, py = click["global_cnn"] if not is_tactical else click["tactical_cnn"]
            if hasattr(self.agent, "cnn_pixel_to_display"):
                cx, cy = self.agent.cnn_pixel_to_display(
                    px,
                    py,
                    (display_w, display_h),
                    obs_size=(obs_size, obs_size),
                )
            else:
                cx = int(px * display_w / float(obs_size))
                cy = int(py * display_h / float(obs_size))
        else:
            spatial_idx = int(click.get("global_spatial", click.get("spatial", 0)))
            grid_size = 32 if is_tactical or hasattr(self.agent, "spatial_resolution") else getattr(
                self.agent, "grid_size", 128
            )
            y_grid, x_grid = divmod(spatial_idx, grid_size)
            scale = display_w / float(grid_size)
            cx = int(x_grid * scale + scale / 2)
            cy = int(y_grid * scale + scale / 2)

        color = (0, 255, 0)
        cv2.line(rendered_frame, (cx - 10, cy), (cx + 10, cy), color, 2)
        cv2.line(rendered_frame, (cx, cy - 10), (cx, cy + 10), color, 2)
        cv2.circle(rendered_frame, (cx, cy), 4, color, 1)
        cv2.circle(rendered_frame, (cx, cy), 2, (255, 255, 255), -1)

    def _render_to_label(self, frame: np.ndarray, label: QLabel, is_tactical: bool = False):
        # frame is upscaled BGR from UI Transformer
        display_h, display_w, _ = frame.shape
        
        # Create a copy for overlays
        rendered_frame = frame.copy()

        # 1. Overlay Action Mask (Global only for now, or adapted if needed)
        if not is_tactical and self.show_mask:
            mask_flat = self.agent.get_action_mask()
            if mask_flat is not None and mask_flat.size == (128 * 128):
                mask_2d = mask_flat.reshape((128, 128))
                mask_resized = cv2.resize(mask_2d.astype(np.uint8), (display_w, display_h),
                                          interpolation=cv2.INTER_NEAREST)
                invalid_area = (mask_resized == 0)
                overlay = rendered_frame.copy()
                overlay[invalid_area] = [0, 0, 200]  # BGR Red
                cv2.addWeighted(overlay, 0.4, rendered_frame, 0.6, 0, rendered_frame)

        # 2. Overlay player / agent click targets
        if self.show_clicks:
            self._draw_click_overlay(rendered_frame, label, is_tactical)

        # Convert to QImage
        rgb_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_frame.data, display_w, display_h, 3 * display_w, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.is_realtime_mode = False
        self.ui_manager.stop_real_time_mode()
        event.accept()

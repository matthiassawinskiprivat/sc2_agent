import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QLabel, QHBoxLayout, QComboBox, 
                             QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from imitation_learning.config_resolver import ConfigResolver
from imitation_learning.recorder import Recorder
from imitation_learning.replay_manager import ReplayManager
from imitation_learning.translator import ReplayTranslator
from imitation_learning.validator import Validator
from src.ui.debug_ui import DebugUI

class GameThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, recorder, level, map_name):
        super().__init__()
        self.recorder = recorder
        self.level = level
        self.map_name = map_name

    def run(self):
        import signal
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        try:
            path = self.recorder.start_recording(self.level, self.map_name)
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal

class TranslationThread(QThread):
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, translator, replay_path, level, config):
        super().__init__()
        self.translator = translator
        self.replay_path = replay_path
        self.level = level
        self.config = config

    def run(self):
        import asyncio
        import traceback
        import signal
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Signal hack: SC2Process tries to register signals which only works in the main thread
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        
        try:
            count = self.translator.translate(self.replay_path, self.level, self.config)
            self.finished.emit(count)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal
            loop.close()

class ValidationThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    ui_ready = pyqtSignal(object, object) # bot, ui_manager

    def __init__(self, validator, dataset_path, level, map_name, config):
        super().__init__()
        self.validator = validator
        self.dataset_path = dataset_path
        self.level = level
        self.map_name = map_name
        self.config = config

    def run(self):
        import signal
        import asyncio
        import traceback
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        try:
            import sc2
            from sc2.main import run_game
            from sc2 import maps
            from sc2.player import Bot, Computer
            from sc2.data import Difficulty
            
            bot, map_path, agent_race, enemy_race, ui_manager = self.validator.start_validation(
                self.dataset_path, self.level, self.map_name, self.config
            )
            
            self.ui_ready.emit(bot, ui_manager)
            
            # Note: game_speed is intentionally omitted as it was causing AttributeErrors 
            # in some environments and defaults to 'Normal' which is safer.
            run_game(
                maps.Map(map_path),
                [
                    Bot(agent_race, bot),
                    Computer(enemy_race, Difficulty.Easy)
                ],
                realtime=True
            )
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal
            loop.close()


class RecordTab(QWidget):
    recording_finished = pyqtSignal()

    def __init__(self, config_resolver: ConfigResolver, recorder: Recorder):
        super().__init__()
        self.resolver = config_resolver
        self.recorder = recorder
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Record gameplay to create raw replays."))
        
        self.level_selector = QComboBox()
        self.map_selector = QComboBox()
        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.on_start_clicked)
        
        self.level_selector.addItems(self.resolver.list_levels())
        self.level_selector.currentTextChanged.connect(self.on_level_changed)
        
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("Level:"))
        form_layout.addWidget(self.level_selector)
        form_layout.addWidget(QLabel("Map:"))
        form_layout.addWidget(self.map_selector)
        
        layout.addLayout(form_layout)
        layout.addWidget(self.start_button)
        layout.addStretch()
        
        # Initial trigger
        if self.level_selector.count() > 0:
            self.on_level_changed(self.level_selector.currentText())

    def on_level_changed(self, level_name):
        self.map_selector.clear()
        maps = self.resolver.list_maps(level_name)
        self.map_selector.addItems(maps)
        self.start_button.setEnabled(len(maps) > 0)

    def on_start_clicked(self):
        level = self.level_selector.currentText()
        map_name = self.map_selector.currentText()
        
        self.start_button.setEnabled(False)
        self.start_button.setText("Recording in progress...")
        
        self.thread = GameThread(self.recorder, level, map_name)
        self.thread.finished.connect(self.on_recording_finished)
        self.thread.error.connect(self.on_recording_error)
        self.thread.start()

    def on_recording_finished(self, path):
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Recording")
        QMessageBox.information(self, "Success", f"Replay saved to:\n{path}")
        self.recording_finished.emit()

    def on_recording_error(self, err):
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Recording")
        QMessageBox.critical(self, "Error", f"Failed to start game:\n{err}")

class TranslateTab(QWidget):
    translation_finished = pyqtSignal()

    def __init__(self, config_resolver: ConfigResolver, replay_manager: ReplayManager, translator: ReplayTranslator):
        super().__init__()
        self.resolver = config_resolver
        self.replay_manager = replay_manager
        self.translator = translator
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Translate raw replays into training data."))
        
        # Level selection for filtering replays
        self.level_selector = QComboBox()
        self.level_selector.addItems(self.resolver.list_levels())
        self.level_selector.currentTextChanged.connect(self.refresh_list)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show Replays for Level:"))
        filter_layout.addWidget(self.level_selector)
        layout.addLayout(filter_layout)

        self.replay_table = QTableWidget(0, 3)
        self.replay_table.setHorizontalHeaderLabels(["Name", "Date", "Size"])
        self.replay_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.replay_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.translate_button = QPushButton("Translate Selected")
        self.translate_button.setEnabled(False)
        self.translate_button.clicked.connect(self.on_translate_clicked)
        self.replay_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.replay_table)
        layout.addWidget(self.translate_button)
        
        self.refresh_list()

    def refresh_list(self):
        level = self.level_selector.currentText()
        replays = self.replay_manager.list_raw_replays(level)
        
        self.replay_table.setRowCount(len(replays))
        for i, r in enumerate(replays):
            self.replay_table.setItem(i, 0, QTableWidgetItem(r["name"]))
            # Store the path in user data
            self.replay_table.item(i, 0).setData(Qt.UserRole, r["path"])
            self.replay_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.replay_table.setItem(i, 2, QTableWidgetItem(r["size"]))

    def on_selection_changed(self):
        self.translate_button.setEnabled(len(self.replay_table.selectedItems()) > 0)

    def on_translate_clicked(self):
        selected_items = self.replay_table.selectedItems()
        if not selected_items: return
        
        row = selected_items[0].row()
        replay_path = self.replay_table.item(row, 0).data(Qt.UserRole)
        level = self.level_selector.currentText()
        config = self.resolver.get_cumulative_config(level)
        
        self.translate_button.setEnabled(False)
        self.translate_button.setText("Translating...")
        
        self.thread = TranslationThread(self.translator, replay_path, level, config)
        self.thread.finished.connect(self.on_translation_finished)
        self.thread.error.connect(self.on_translation_error)
        self.thread.start()

    def on_translation_finished(self, count):
        self.translate_button.setEnabled(True)
        self.translate_button.setText("Translate Selected")
        QMessageBox.information(self, "Success", f"Translation complete!\nGenerated {count} training examples.")
        self.translation_finished.emit()

    def on_translation_error(self, err):
        self.translate_button.setEnabled(True)
        self.translate_button.setText("Translate Selected")
        QMessageBox.critical(self, "Error", f"Translation failed:\n{err}")

class VerifyTab(QWidget):
    def __init__(self, config_resolver: ConfigResolver, replay_manager: ReplayManager, validator: Validator):
        super().__init__()
        self.resolver = config_resolver
        self.replay_manager = replay_manager
        self.validator = validator
        self.debug_window = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Verify translated data via visual playback."))
        
        # Level selection for filtering
        self.level_selector = QComboBox()
        self.level_selector.addItems(self.resolver.list_levels())
        self.level_selector.currentTextChanged.connect(self.refresh_list)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show Data for Level:"))
        filter_layout.addWidget(self.level_selector)
        layout.addLayout(filter_layout)

        self.translated_table = QTableWidget(0, 3)
        self.translated_table.setHorizontalHeaderLabels(["Name", "Date", "Size"])
        self.translated_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.translated_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.verify_button = QPushButton("Verify (Watch)")
        self.verify_button.setEnabled(False)
        self.verify_button.clicked.connect(self.on_verify_clicked)
        self.translated_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.translated_table)
        layout.addWidget(self.verify_button)
        
        self.refresh_list()

    def refresh_list(self):
        level = self.level_selector.currentText()
        replays = self.replay_manager.list_translated_replays(level)
        
        self.translated_table.setRowCount(len(replays))
        for i, r in enumerate(replays):
            self.translated_table.setItem(i, 0, QTableWidgetItem(r["name"]))
            self.translated_table.item(i, 0).setData(Qt.UserRole, r["path"])
            self.translated_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.translated_table.setItem(i, 2, QTableWidgetItem(r["size"]))

    def on_selection_changed(self):
        self.verify_button.setEnabled(len(self.translated_table.selectedItems()) > 0)

    def on_verify_clicked(self):
        selected_items = self.translated_table.selectedItems()
        if not selected_items: return
        
        row = selected_items[0].row()
        dataset_path = self.translated_table.item(row, 0).data(Qt.UserRole)
        level = self.level_selector.currentText()
        config = self.resolver.get_cumulative_config(level)
        
        maps = self.resolver.list_maps(level)
        if not maps:
            QMessageBox.warning(self, "Warning", f"No maps found for level {level}. Cannot launch validation.")
            return
        
        map_name = maps[0] 
        filename = os.path.basename(dataset_path)
        for m in maps:
            if m in filename:
                map_name = m
                break

        self.verify_button.setEnabled(False)
        self.verify_button.setText("Validation running...")
        
        # Start game in thread
        self.thread = ValidationThread(self.validator, dataset_path, level, map_name, config)
        self.thread.ui_ready.connect(self.show_debug_ui)
        self.thread.finished.connect(self.on_validation_finished)
        self.thread.error.connect(self.on_validation_error)
        self.thread.start()

    def show_debug_ui(self, bot, ui_manager):
        # This runs in the main thread
        self.debug_window = DebugUI(bot.mock_agent, ui_manager)
        self.debug_window.setWindowTitle(f"Validation Playback")
        self.debug_window.show()

    def on_validation_finished(self):
        self.verify_button.setEnabled(True)
        self.verify_button.setText("Verify (Watch)")
        if self.debug_window:
            self.debug_window.close()

    def on_validation_error(self, err):
        self.verify_button.setEnabled(True)
        self.verify_button.setText("Verify (Watch)")
        QMessageBox.critical(self, "Error", f"Validation failed:\n{err}")
        if self.debug_window:
            self.debug_window.close()

class ImitationLearningWorkstation(QMainWindow):
    def __init__(self, project_root: str = "."):
        super().__init__()
        self.resolver = ConfigResolver(project_root)
        self.recorder = Recorder(project_root)
        self.replay_manager = ReplayManager(project_root)
        self.translator = ReplayTranslator(project_root)
        self.validator = Validator(project_root)
        
        self.setWindowTitle("SC2 Imitation Learning Workstation")
        self.resize(1000, 700)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.record_tab = RecordTab(self.resolver, self.recorder)
        self.translate_tab = TranslateTab(self.resolver, self.replay_manager, self.translator)
        self.verify_tab = VerifyTab(self.resolver, self.replay_manager, self.validator)
        
        # Connect signals
        self.record_tab.recording_finished.connect(self.translate_tab.refresh_list)
        self.translate_tab.translation_finished.connect(self.verify_tab.refresh_list)
        
        self.tabs.addTab(self.record_tab, "Record")
        self.tabs.addTab(self.translate_tab, "Translate")
        self.tabs.addTab(self.verify_tab, "Verify")

def main():
    app = QApplication(sys.argv)
    window = ImitationLearningWorkstation()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

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

class GameThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, recorder, level, map_name):
        super().__init__()
        self.recorder = recorder
        self.level = level
        self.map_name = map_name

    def run(self):
        try:
            path = self.recorder.start_recording(self.level, self.map_name)
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))

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
    def __init__(self, config_resolver: ConfigResolver, replay_manager: ReplayManager):
        super().__init__()
        self.resolver = config_resolver
        self.replay_manager = replay_manager
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
            self.replay_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.replay_table.setItem(i, 2, QTableWidgetItem(r["size"]))

    def on_selection_changed(self):
        self.translate_button.setEnabled(len(self.replay_table.selectedItems()) > 0)

class VerifyTab(QWidget):
    def __init__(self, config_resolver: ConfigResolver, replay_manager: ReplayManager):
        super().__init__()
        self.resolver = config_resolver
        self.replay_manager = replay_manager
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Verify translated data via visual playback."))
        
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
            self.translated_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.translated_table.setItem(i, 2, QTableWidgetItem(r["size"]))

    def on_selection_changed(self):
        self.verify_button.setEnabled(len(self.translated_table.selectedItems()) > 0)

class ImitationLearningWorkstation(QMainWindow):
    def __init__(self, project_root: str = "."):
        super().__init__()
        self.resolver = ConfigResolver(project_root)
        self.recorder = Recorder(project_root)
        self.replay_manager = ReplayManager(project_root)
        
        self.setWindowTitle("SC2 Imitation Learning Workstation")
        self.resize(1000, 700)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.record_tab = RecordTab(self.resolver, self.recorder)
        self.translate_tab = TranslateTab(self.resolver, self.replay_manager)
        self.verify_tab = VerifyTab(self.resolver, self.replay_manager)
        
        # Connect signals
        self.record_tab.recording_finished.connect(self.translate_tab.refresh_list)
        
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

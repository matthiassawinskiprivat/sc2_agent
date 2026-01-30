import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QLabel, QHBoxLayout, QComboBox, 
                             QPushButton, QListWidget, QTableWidget)
from PyQt5.QtCore import Qt
from imitation_learning.config_resolver import ConfigResolver

class RecordTab(QWidget):
    def __init__(self, config_resolver: ConfigResolver):
        super().__init__()
        self.resolver = config_resolver
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Record gameplay to create raw replays."))
        
        self.level_selector = QComboBox()
        self.map_selector = QComboBox()
        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        
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

class TranslateTab(QWidget):
    def __init__(self, config_resolver: ConfigResolver):
        super().__init__()
        self.resolver = config_resolver
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Translate raw replays into training data."))
        
        self.replay_list = QTableWidget()
        self.translate_button = QPushButton("Translate Selected")
        self.translate_button.setEnabled(False)
        
        layout.addWidget(self.replay_list)
        layout.addWidget(self.translate_button)

class VerifyTab(QWidget):
    def __init__(self, config_resolver: ConfigResolver):
        super().__init__()
        self.resolver = config_resolver
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Verify translated data via visual playback."))
        
        self.translated_list = QTableWidget()
        self.verify_button = QPushButton("Verify (Watch)")
        self.verify_button.setEnabled(False)
        
        layout.addWidget(self.translated_list)
        layout.addWidget(self.verify_button)

class ImitationLearningWorkstation(QMainWindow):
    def __init__(self, project_root: str = "."):
        super().__init__()
        self.resolver = ConfigResolver(project_root)
        self.setWindowTitle("SC2 Imitation Learning Workstation")
        self.resize(800, 600)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.record_tab = RecordTab(self.resolver)
        self.translate_tab = TranslateTab(self.resolver)
        self.verify_tab = VerifyTab(self.resolver)
        
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

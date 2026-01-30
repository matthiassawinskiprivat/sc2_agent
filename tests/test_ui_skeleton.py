import sys
import pytest
from PyQt5.QtWidgets import QApplication, QTabWidget
from imitation_learning.app import ImitationLearningWorkstation

@pytest.fixture
def app(qtbot):
    # Ensure a QApplication instance exists
    test_app = QApplication.instance()
    if test_app is None:
        test_app = QApplication(sys.argv)
    
    window = ImitationLearningWorkstation()
    qtbot.addWidget(window)
    return window

def test_main_window_structure(app):
    assert app.windowTitle() == "SC2 Imitation Learning Workstation"
    
    # Check for tab widget
    tab_widget = app.findChild(QTabWidget)
    assert tab_widget is not None
    assert tab_widget.count() == 3
    
    # Check tab labels
    assert tab_widget.tabText(0) == "Record"
    assert tab_widget.tabText(1) == "Translate"
    assert tab_widget.tabText(2) == "Verify"

def test_level_and_map_selectors(app, qtbot):
    # Mock data should be handled by ConfigResolver (we can patch it if needed)
    # For now let's assume it picks up the real project structure
    
    level_selector = app.record_tab.level_selector
    map_selector = app.record_tab.map_selector
    
    assert level_selector.count() > 0
    
    # Select a known level (that we know has maps)
    level_name = "level_0" 
    index = level_selector.findText(level_name)
    if index >= 0:
        level_selector.setCurrentIndex(index)
        
        # Verify maps are updated
        assert map_selector.count() > 0
        assert app.record_tab.start_button.isEnabled()

def test_replay_list_population(app):
    # Verify the table has headers
    assert app.translate_tab.replay_table.columnCount() == 3
    assert app.translate_tab.replay_table.horizontalHeaderItem(0).text() == "Name"



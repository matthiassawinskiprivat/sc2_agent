from sc2.data import Race

from src.ui.debug_ui import MODE_FREE_PLAY, MODE_TRAINING_LEVEL, race_from_ui_label


def test_race_from_ui_label_maps_all_options():
    assert race_from_ui_label("Terran") == Race.Terran
    assert race_from_ui_label("Zerg") == Race.Zerg
    assert race_from_ui_label("Protoss") == Race.Protoss
    assert race_from_ui_label("Random") == Race.Random


def test_debug_ui_mode_constants():
    assert MODE_FREE_PLAY == "free_play"
    assert MODE_TRAINING_LEVEL == "training_level"

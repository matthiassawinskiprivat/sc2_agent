"""Targeted SC2 process-tree cleanup (worker-safe, no global kill)."""
from unittest.mock import MagicMock, patch

from src.utils.sc2_cleanup import count_sc2_processes, kill_sc2_process_tree


def test_kill_sc2_process_tree_none_pid():
    assert kill_sc2_process_tree(None) is False


def test_kill_sc2_process_tree_kills_children_and_root():
    child = MagicMock()
    child.pid = 200
    child.name.return_value = "SC2_x64.exe"
    child.is_running.return_value = False

    root = MagicMock()
    root.pid = 100
    root.name.return_value = "SC2_x64.exe"
    root.children.return_value = [child]
    root.is_running.return_value = False

    with patch("src.utils.sc2_cleanup.psutil.Process", return_value=root) as mock_proc:
        assert kill_sc2_process_tree(100, wait_sec=0.0) is True
        mock_proc.assert_called_once_with(100)
        child.kill.assert_called_once()
        root.kill.assert_called_once()


def test_count_sc2_processes_filters_by_name():
    sc2 = MagicMock()
    sc2.info = {"pid": 1, "name": "SC2_x64.exe"}
    other = MagicMock()
    other.info = {"pid": 2, "name": "python.exe"}

    with patch("src.utils.sc2_cleanup.psutil.process_iter", return_value=[sc2, other]):
        assert count_sc2_processes() == 1

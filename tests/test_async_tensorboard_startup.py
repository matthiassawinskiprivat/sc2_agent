"""TensorBoard auto-start for async universal training."""
from unittest.mock import MagicMock, patch

from train.async_train_universal import _start_tensorboard, free_port, refresh_tensorboard


@patch("train.async_train_universal._tensorboard_port_open", return_value=False)
@patch("train.async_train_universal.subprocess.Popen")
@patch("train.async_train_universal.time.sleep")
def test_async_start_tensorboard(mock_sleep, mock_popen, _mock_port, tmp_path):
    log_base = tmp_path / "logs_universal"
    log_base.mkdir()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_popen.return_value = mock_proc

    with patch(
        "train.async_train_universal._tensorboard_port_open",
        side_effect=[False, True],
    ):
        _start_tensorboard(str(log_base), port=6006)

    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert "tensorboard.main" in cmd
    assert "--logdir" in cmd
    assert str(log_base.resolve()) in cmd
    assert "6006" in cmd
    assert "--bind_all" in cmd
    assert "--reload_interval" in cmd


@patch("train.async_train_universal.subprocess.Popen")
@patch("train.async_train_universal._tensorboard_port_open", return_value=True)
def test_async_start_tensorboard_skips_when_port_busy(mock_port, mock_popen, tmp_path):
    _start_tensorboard(str(tmp_path / "logs_universal"), port=6006)
    mock_popen.assert_not_called()


@patch("train.async_train_universal.psutil.Process")
@patch("train.async_train_universal.psutil.net_connections")
def test_free_port_terminates_listener(mock_net_conns, mock_process_cls):
    conn = MagicMock()
    conn.laddr.port = 6006
    conn.status = "LISTEN"
    conn.pid = 4242
    mock_net_conns.return_value = [conn]
    proc = MagicMock()
    mock_process_cls.return_value = proc

    n = free_port(6006)

    assert n == 1
    proc.terminate.assert_called_once()


@patch("train.async_train_universal._start_tensorboard")
@patch("train.async_train_universal.free_port")
def test_refresh_tensorboard_restarts_between_levels(mock_free, mock_start, tmp_path):
    refresh_tensorboard(str(tmp_path / "logs_universal"), 6006, active_level="level_attack_1")
    mock_free.assert_called_once_with(6006)
    mock_start.assert_called_once()


@patch("train.async_train_universal._start_tensorboard")
@patch("train.async_train_universal.free_port")
def test_refresh_tensorboard_skipped_when_disabled(mock_free, mock_start):
    import os

    os.environ["SC2_NO_TENSORBOARD_LEVEL_REFRESH"] = "1"
    try:
        refresh_tensorboard(active_level="level_attack_1")
        mock_free.assert_not_called()
        mock_start.assert_not_called()
    finally:
        os.environ.pop("SC2_NO_TENSORBOARD_LEVEL_REFRESH", None)

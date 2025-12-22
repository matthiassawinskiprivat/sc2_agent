import multiprocessing
import pytest
from unittest.mock import MagicMock, patch
from train.game_process import GameProcess

class TestGameProcessLogging:
    
    @patch("train.game_process.logger")
    @patch("train.game_process.run_game")
    @patch("train.game_process.Bot")
    @patch("train.game_process.SC2Bot")
    def test_game_process_lifecycle_logging(self, mock_sc2bot, mock_bot, mock_run_game, mock_logger):
        """
        Test that GameProcess logs critical lifecycle events.
        """
        mock_conn = MagicMock(spec=multiprocessing.connection.Connection)
        game_config = {
            "map_name": "00_32x32_No_Middle.SC2Map",
            "realtime": False,
            "level_name": "level_0"
        }
        
        process = GameProcess(game_config, mock_conn)
        
        # Simulate running the process
        # We need to mock _resolve_map to avoid file system dependency or complex mocking
        with patch.object(process, "_resolve_map", return_value=MagicMock()) as mock_resolve:
            process.run()
            
        # Verify start log
        start_log_found = False
        bot_init_found = False
        run_game_start_found = False
        run_game_end_found = False
        finish_log_found = False

        for call in mock_logger.info.call_args_list:
            msg = call[0][0]
            print(f"LOG INFO: {msg}")
            if "started" in msg:
                start_log_found = True
            if "Initializing SC2Bot" in msg:
                bot_init_found = True
            if "Calling run_game" in msg:
                run_game_start_found = True
            if "run_game returned" in msg:
                run_game_end_found = True
            if "finished" in msg:
                finish_log_found = True
        
        for call in mock_logger.error.call_args_list:
             print(f"LOG ERROR: {call[0][0]}")

        assert start_log_found, "GameProcess start should be logged"
        assert bot_init_found, "SC2Bot initialization should be logged"
        assert run_game_start_found, "Pre-run_game call should be logged"
        assert run_game_end_found, "Post-run_game call should be logged"
        assert finish_log_found, "GameProcess finish should be logged"
        
        # Verify close was called
        mock_conn.close.assert_called_once()

    @patch("train.game_process.logger")
    def test_game_process_map_resolution_logging(self, mock_logger):
        """
        Test detailed logging during map resolution.
        """
        mock_conn = MagicMock()
        game_config = {
            "map_name": "NonExistentMap.SC2Map",
            "realtime": False,
            "level_name": "level_0"
        }
        
        process = GameProcess(game_config, mock_conn)
        
        # We expect a FileNotFoundError, but we are testing logging before that
        with pytest.raises(FileNotFoundError):
            process._resolve_map(game_config["map_name"])
            
        # Verify resolution start log
        resolve_call_found = False
        for call in mock_logger.info.call_args_list:
            if "_resolve_map called" in call[0][0]:
                resolve_call_found = True
                break
        assert resolve_call_found, "Map resolution start should be logged"

        # Verify path construction log
        path_log_found = False
        for call in mock_logger.info.call_args_list:
            if "Constructed map_path" in call[0][0]:
                path_log_found = True
                break
        assert path_log_found, "Map path construction should be logged"

        # Verify error log
        error_log_found = False
        for call in mock_logger.error.call_args_list:
            if "Map not found" in call[0][0]:
                error_log_found = True
                break
        assert error_log_found, "Map not found error should be logged"


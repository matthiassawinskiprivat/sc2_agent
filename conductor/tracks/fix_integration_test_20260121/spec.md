# Track: Fix Level Drop Integration Test

## Context
The test `tests/integration_test_level_drop_init.py` attempts to run a short training session by creating a temporary configuration file `config/level_drop_0_tmp.yaml`. However, the `PipelineManager` is instructed to load `level_drop_0`, which causes it to read the production config `config/level_drop_0.yaml` instead, ignoring the temporary file. This leads to the test running with production settings (long training steps).

## Goals
-   Fix `tests/integration_test_level_drop_init.py` to correctly use the modified (short) configuration.
-   Stop creating `config/level_drop_0_tmp.yaml` on disk.
-   Use `unittest.mock` to inject the modified configuration when `PipelineManager` tries to read `config/level_drop_0.yaml`.
-   Ensure the test passes and actually runs the short training loop.

## Requirements
-   Do NOT modify production code (`train/train_pipeline.py`, `src/utils/config_manager.py`).
-   Do NOT modify production config (`config/level_drop_0.yaml`).
-   Only modify `tests/integration_test_level_drop_init.py`.

## Plan
1.  Remove file I/O for `level_drop_0_tmp.yaml`.
2.  Import `unittest.mock.patch` and `mock_open`.
3.  Define a `mocked_open` side effect that:
    *   Intercepts reads to `config/level_drop_0.yaml`.
    *   Returns a `mock_open` object with the modified configuration (dumped as YAML string).
    *   Delegates all other calls to the real `builtins.open`.
4.  Wrap the `manager._train_level` call with `patch('builtins.open', side_effect=mocked_open)`.

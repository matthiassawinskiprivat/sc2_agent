# Plan: Fix Level Drop Integration Test

## Steps
1.  **Read** the current test file `tests/integration_test_level_drop_init.py` to preserve imports and setup logic.
2.  **Rewrite** the test file:
    *   Add imports: `from unittest.mock import patch, mock_open`, `import builtins`.
    *   Keep the logic that loads and modifies the config dict (`level_conf`).
    *   Remove `with open("config/level_drop_0_tmp.yaml", "w")...`.
    *   Implement `mocked_open` function.
    *   Use context manager: `with patch('builtins.open', side_effect=mocked_open):` around the `manager._train_level` call.
    *   Pass `level_drop_0` (not `tmp`) to `_train_level`.
3.  **Run** the test to verify it passes and completes quickly (confirming it used the 100-step config).

## Todo List
- [ ] Read `tests/integration_test_level_drop_init.py`.
- [ ] Rewrite `tests/integration_test_level_drop_init.py`.
- [ ] Run `python tests/integration_test_level_drop_init.py`.

# Plan - Fix CTF Single Env Termination

- [x] Task: Create a reproduction script `tests/test_ctf_single_termination_repro.py` specifically for `CaptureTheFlagSingleEnv` that mocks the bot and unit list to simulate the "trainee lost all units" scenario.
- [x] Task: Update `train/envs/capture_the_flag_single_env.py` to include the `_has_combat_units` check (or equivalent logic) and trigger termination if true.
- [x] Task: Verify the fix using the reproduction script.
- [x] Task: Run `tests/test_ctf_end_conditions_logic.py` to ensure no regressions in existing logic.
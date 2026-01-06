# Specification: Fix Parallel Process Termination

## Overview
When running parallel training (`num_envs > 1`), individual environments finish episodes at different times. When an environment finishes, it calls `reset()`, which triggers a cleanup of the previous session. Currently, `SingleClientSC2Manager.close()` calls `kill_switch_sc2()`, which aggressively terminates ALL StarCraft II processes on the system. This kills the processes owned by other parallel environments, causing them to crash with `BrokenPipeError` or `EOFError`.

## Functional Requirements
1.  **Process-Specific Termination:**
    *   Modify `SingleClientSC2Manager.close()` to only terminate the StarCraft II process it specifically manages.
    *   Do NOT call `kill_switch_sc2()` (the system-wide nuke) during per-episode cleanup.
2.  **Robust Cleanup:**
    *   Ensure that the managed process is reliably terminated to prevent resource leaks.
    *   Use the `SC2Process.pid` if available to perform a targeted kill if `__aexit__` is insufficient.
3.  **Maintain Global Cleanup Availability:**
    *   `kill_switch_sc2()` should still be available for full pipeline startup/shutdown, but not for per-worker resets.

## Acceptance Criteria
-   [ ] `train_pipeline` with `num_envs: 2` survives the transition where one environment resets while the other is still stepping.
-   [ ] Worker logs no longer show `Cannot write to closing transport` or abrupt termination during a sibling's reset.
-   [ ] No zombie StarCraft II processes are left behind after a worker is closed (verified via process monitoring).

## Out of Scope
-   Optimizing SC2 process startup speed.
-   Changing the internal port allocation of `burnysc2`.

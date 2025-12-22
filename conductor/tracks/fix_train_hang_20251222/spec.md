# Specification - Fix premature train_pipeline termination/hang

## Overview
The `train_pipeline.py` occasionally stops executing during training sessions without reporting any errors. While "soft/hard resets" occur normally for a while, eventually the process hangs, leaving StarCraft 2 game windows open and unresponsive. This behavior prevents the pipeline from reaching the configured completion criteria (steps or evaluation) defined in `level_0.yaml`.

## Problem Description
- **Symptoms:** Pipeline stops progressing; SC2 windows remain open; No error logs are generated.
- **Frequency:** Randomly after varying durations.
- **Hypothesis:** A deadlock or a hanging process during the environment synchronization or the "hard reset" phase, where the communication between the Python process and the SC2 client is lost or blocked.

## Functional Requirements
1. **Enhanced Instrumentation:** Add high-resolution logging around environment `reset()`, `step()`, and process management calls in `train_pipeline.py`, `sync_vec_env.py`, and `game_process.py`.
2. **Watchdog Timer:** Implement a watchdog mechanism that monitors critical loop execution and logs a stack trace or alert if the pipeline hangs for longer than a specified threshold.
3. **Diagnosis:** Identify the exact line of code or communication block where the hang occurs.
4. **Termination Integrity:** Ensure the pipeline only stops when the `level_0.yaml` criteria (max steps or evaluation success) are met.
5. **Reliable Cleanup:** Ensure that if the pipeline does stop (or even if it hangs), SC2 processes are reliably terminated.

## Non-Functional Requirements
- **Observability:** Logging should include timestamps and state markers (e.g., "Entering hard reset", "Waiting for game process").

## Acceptance Criteria
1. The training pipeline runs to completion (based on configuration) in multiple test runs without hanging.
2. Logs clearly show the sequence of events leading up to every reset and termination.
3. The watchdog timer successfully detects hangs and provides diagnostic information (if the hang persists).
4. No StarCraft 2 processes are left orphaned after the training pipeline exits.

## Out of Scope
- Optimizing the performance of the RL model.
- Changing the game logic within the SC2 bot.

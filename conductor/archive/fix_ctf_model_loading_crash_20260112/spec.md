# Specification: Fix Training Pipeline Model Loading Crash

## Overview
The training pipeline (`train_pipeline.py`) is crashing with a `BrokenPipeError` / `EOFError` when attempting to train or resume a level. The user reports that this issue appeared after various AI-generated code changes. The main process logs indicate a communication failure with worker processes, pointing to a crash within the worker environment itself. Relevant worker logs exist and contain error details.

## Functional Requirements
1.  **Root Cause Analysis:**
    *   Analyze the contents of `logs/` (specifically the worker log containing the error) to identify the specific exception causing the worker to crash.
    *   Correlate the error with recent code changes or model structure definitions.
2.  **Crash Fix:**
    *   Resolve the underlying issue causing the worker process to terminate (e.g., shape mismatch, serialization/pickling issue, incorrect environment configuration, or resource exhaustion).
    *   Ensure the fix handles model loading and state restoration correctly.
3.  **Stability:**
    *   Ensure the training pipeline can successfully start and (if applicable) resume from saved models without immediate termination.

## Acceptance Criteria
*   **Pipeline Runs:** Executing `train/train_pipeline.py` (or the relevant command) successfully starts the training loop without throwing `BrokenPipeError` or `EOFError`.
*   **Models Load:** If an existing model is present, the system loads it correctly (or handles the mismatch gracefully if that's the intended behavior) instead of crashing.
*   **Logs Clear:** Worker logs do not report fatal exceptions during the startup/reset phase.

## Out of Scope
*   General refactoring of the entire agent (unless necessary to fix the specific crash).
*   Optimizing training performance (focus is on stability/crashing).

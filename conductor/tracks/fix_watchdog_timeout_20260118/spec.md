# Specification: Fix Watchdog Timeout during Training Optimization

## Overview
The training pipeline utilizes a `Watchdog` class to detect hanging processes. It has a default timeout of 300 seconds (5 minutes). This watchdog is kicked (reset) via a callback during the rollout (data collection) phase. However, Stable Baselines 3's `train()` method (optimization phase) does not trigger callbacks.

For complex models (CNN with 44 input channels) running on CPU, the optimization phase for a single iteration can exceed 300 seconds. This causes the Watchdog to incorrectly identify the slow training as a "hang" and terminate the process.

This track will make the watchdog timeout configurable and increase the default to a value safe for CPU training (e.g., 1 hour), ensuring robustness against slow hardware.

## Functional Requirements
1.  **Configurable Timeout:** The `train_pipeline.py` script must read a `watchdog_timeout` parameter from the training configuration (yaml files).
2.  **Increased Default:** If no configuration is provided, the default timeout should be increased significantly (e.g., to 3600 seconds) to prevent false positives during valid computation.
3.  **Graceful Fallback:** Ensure that valid hangs (actual deadlocks) are still eventually caught, but with a much wider margin.

## Non-Functional Requirements
1.  **Minimal Code Change:** Focus changes on `train_pipeline.py` initialization logic.
2.  **No Logic Change:** The actual training or watchdog logic remains the same; only the threshold parameter is adjusted.

## Acceptance Criteria
1.  `train_pipeline.py` respects a `watchdog_timeout` setting in `level_*.yaml` (under `training` section).
2.  The default timeout in `train_pipeline.py` is updated to 3600.0 (1 hour).
3.  A test or validation run confirms the new timeout value is used.

## Out of Scope
*   Modifying SB3 internal code to support optimization callbacks.
*   Optimizing the neural network to run faster (that was handled in the previous track).

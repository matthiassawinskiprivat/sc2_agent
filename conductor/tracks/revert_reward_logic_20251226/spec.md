# Specification: Revert Reward Function to Master Branch Logic

## 1. Overview
The current reward function in `src/sc2_bot.py` has been identified as less effective than the version in the `master` branch. The user reports that the agent learned significantly better with the previous logic. The goal is to retrieve the `_calculate_reward` method (and any associated state management like `previous_distance`) from the `master` branch and restore it in the current codebase.

## 2. Problem Description
*   **Symptom:** Degraded learning performance after refactoring.
*   **Root Cause:** The reward shaping formula was altered during refactoring.
*   **Goal:** Revert to the known-good logic from `master`.

## 3. Functional Requirements
*   **Retrieve Logic:** The `_calculate_reward` method must be extracted from `src/sc2_bot.py` in the `master` branch.
*   **Restore Logic:** The current `_calculate_reward` in `src/sc2_bot.py` must be replaced with the extracted code.
*   **State Management:** The handling of `previous_distance` (initialization and updates) must match the `master` branch implementation exactly.

## 4. Acceptance Criteria
*   **Code Match:** The `_calculate_reward` function in the current branch matches the logic of the `master` branch.
*   **State Handling:** `previous_distance` is reset and updated exactly as it was in `master`.
*   **Verification:** A test or manual check confirms the code is identical to the target logic.

## 5. Out of Scope
*   Modifying the reward function beyond reverting to the `master` version.
*   Tuning hyperparameters of the reward function (unless they were part of the `master` code).

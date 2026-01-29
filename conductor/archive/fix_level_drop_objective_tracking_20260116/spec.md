# Specification: Fix Level Drop Objective Tracking

The `LevelDropBot` currently fails to identify the enemy main structure (objective) at the start of training episodes, resulting in a "No enemy main structure found!" warning and breaking the command objective layer visualization. This prevents the agent from receiving spatial guidance toward the target.

## Overview
This track focuses on ensuring the `LevelDropBot` reliably identifies the target objective and correctly populates the `command_target` in the observation, which is rendered on the command objective layer.

## Functional Requirements
- **Robust Objective Identification:** Modify `LevelDropBot` to continuously scan for enemy main structures (Command Center, Hatchery, Nexus, etc.) until one is found, rather than only checking during the initial `reset_game_state`.
- **Observation Integration:** Ensure `raw_obs['command_target']` is populated with the objective's position as soon as it is identified.
- **Command Objective Layer Visibility:** Confirm that the identified objective is correctly rendered on the `command_objective` layer (Layer 41) to provide spatial guidance to the agent.

## Non-Functional Requirements
- **Logging:** Maintain clear log messages when the objective is found and if it remains missing after several steps.
- **Performance:** The scan should be efficient and not cause significant overhead in the training loop.

## Acceptance Criteria
- [ ] The "No enemy main structure found!" warning no longer appears after the first few steps of the episode.
- [ ] `self._objective_pos` is correctly set to the position of an enemy main structure.
- [ ] `raw_obs['command_target']` contains the objective position in every observation after identification.
- [ ] A test case confirms that `LevelDropBot` can recover and find the objective even if it's not present at frame 0.

## Out of Scope
- Modifying reward calculations (load/unload/distance).
- Adding new training levels or environments.
- Fixing unrelated bot behaviors.

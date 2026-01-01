# Track Specification: Level 3 Environment - Multi-Zergling Control

## Overview
The goal is to implement `Level3Env`, an evolution of `Level0Env`. While `Level0Env` tasked a single zergling to find a single beacon, `Level3Env` introduces multi-agent control. The agent must successfully direct multiple zerglings to collect multiple beacons. This environment serves as a stepping stone towards complex micro-management.

## Core Requirements

### 1. Entity Configuration
*   **Dynamic Count:** The environment must support a configurable number of units, defined by the `targets` parameter in the YAML config (e.g., `level_3.yaml`).
*   **Initial State:**
    *   Spawn `N` Zerglings (where `N = targets`).
    *   Spawn `N` Beacons.
    *   Locations for all entities should be random (within valid map bounds).

### 2. Success & Episode Logic
*   **Sequential Collection:** When a zergling touches a beacon, both that specific zergling and the beacon are "collected" and removed from the map.
*   **Episode Success:** The episode ends successfully **only** when all beacons have been collected (i.e., all zerglings have reached their targets).
*   **State Updates:**
    *   **Visuals:** Removed units/beacons must immediately disappear from the observation layers (pixel values reset to 0).
    *   **Logic:** Removed units must be excluded from future processing in the current episode.

### 3. Reward Structure
The reward system mirrors `Level0Env` but normalized for multiple targets:
*   **Success Reward:** `(success_reward / targets)` is granted immediately upon *each* individual collection event.
*   **Distance Reward/Penalty:** Calculated as the **average** distance of all *remaining* active zerglings to their nearest remaining beacons. This average is then used to compute the step reward/penalty.

### 4. Control Scheme (Sequential)
*   **Iterative Control:** The environment operates using a sequential control loop within a single environment step.
    *   For `N` active zerglings, the agent is queried `N` times.
    *   Cycle: Observe State -> Query Agent -> Execute Action for Zergling `i` -> Update Partial State -> Repeat for Zergling `i+1`.
    *   The `command_objective` layer (or equivalent) will be enabled via config to support this if necessary, though the primary mechanism is simply iterating through available units.
    *   **Action Masking:** Actions for units that have already acted or are removed must be effectively masked or skipped.

## Technical Constraints
*   **No Regression:** `Level0Env` must remain completely unchanged.
*   **Configuration:** All parameters (targets, grid size, rewards) must be loaded from `config/level_3.yaml`.
*   **Class Structure:** `Level3Env` should likely inherit from `Level0Env` or share a common base, overriding `reset`, `step`, and reward calculation methods.

## Acceptance Criteria
1.  **Map Generation:** `Level3Env` correctly spawns 2 Zerglings and 2 Beacons (as per current config).
2.  **Collection Logic:** Verifiable removal of zergling+beacon pairs upon contact.
3.  **Completion:** Episode terminates with "Success" status only after both pairs are collected.
4.  **Reward Function:** Total accumulated reward for a perfect run equals `success_reward` (plus/minus distance terms), properly split across the events.
5.  **Independence:** Running `Level0Env` tests still passes 100%.

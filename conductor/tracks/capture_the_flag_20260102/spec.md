# Specification: CTF Environment with Self-Play (CaptureTheFlagEnv)

## 1. Overview
This track implements a new training environment (`CaptureTheFlagEnv`) for a "Capture the Flag" (CTF) scenario. Two agents, both controlled by the same neural network (self-play), compete to destroy the opponent's "Flag" (Pylon) while defending their own. The environment introduces resource-based unit spawning, explicit objective layers, and a competitive reward structure.

## 2. Functional Requirements

### 2.1. Environment Configuration
-   **Class Name:** `CaptureTheFlagEnv` (inheriting from `Level0Env` or a new `MultiAgentEnv` base).
-   **Parameters:**
    -   `minerals`: Starting mineral count for spawning.
    -   `vespene`: Starting vespene count for spawning.
    -   `flag_distance_min`: Minimum distance between the two Pylons.
    -   `unit_pool`: List of allowable unit types (e.g., `["ZERGLING", "ROACH", "MARINE"]`).
    -   `rewards`: Specific reward/penalty values (see 2.4).

### 2.2. Map & Setup
-   **Flags:** Two Pylons are spawned at random locations, separated by at least `flag_distance_min`.
    -   **Attributes:** Health and Shields reduced to 10 points each.
    -   **Visibility:** Always visible on the `command_objective` observation layer.
-   **Unit Spawning:**
    -   Units are chosen randomly from `unit_pool`.
    -   Costs are deducted from the configured `minerals`/`vespene`.
    -   Spawning continues until resources are exhausted for both players.
    -   **Placement:** All units spawn in a cluster near their respective Pylons.

### 2.3. Agent Control (Self-Play)
-   **Control Scheme:** Serial Unit Control (Same as Level 3). The agent controls one unit at a time.
-   **Opponent:** The opponent is controlled by the **same** model (Shared Policy).
-   **Infrastructure:** The environment must support two simultaneous SC2 clients (Host & Join).
-   **Synchronization:** Steps must be synchronized so both agents act before the physics advances.
-   **Layer Specifics:** When the agent takes a step for a specific unit:
    -   `command_assignment` layer must highlight the **active own unit** being controlled.
    -   `command_objective` layer must highlight the **enemy Pylon** (target).

### 2.4. Reward Structure
-   **Step Reward:** Small reward/penalty for the active unit's movement (consistent with previous levels).
-   **Combat:**
    -   Positive reward for damaging enemy units.
    -   Small negative reward for taking damage.
-   **Objective:**
    -   **High Reward:** Damaging the enemy Pylon.
    -   **High Penalty:** Taking damage to the own Pylon.
    -   **Max Reward/Penalty:** Winning (Destroying Enemy Pylon) or Losing (Own Pylon Destroyed).
-   **Termination:** Game ends on Pylon destruction or `max_steps`.

### 2.5. Observation Space
-   **Layers:** 7 Layers (Standard 6 + `command_objective`).
    -   Layer 0-5: Standard Combat/State layers (as defined in `CORRECTED_ARCHITECTURE.md`).
    -   Layer 6: `command_objective` (Explicitly shows the Pylons).
-   **Propagation:** This 7-layer structure is standardized across the pipeline, ensuring `Level0` configurations (which already include `command_objective`) are respected.

## 3. Technical Implementation
-   **New Environment:** `train/envs/capture_the_flag_env.py`.
-   **Config:** `config/capture_the_flag.yaml`.
-   **Multi-Client Support:** Implement `MultiClientEnv` infrastructure to handle two SC2 processes.
-   **Transformer Update:** Ensure `SharedObservationTransformer` correctly handles the 7-layer stack and Pylon visualization.

## 4. Acceptance Criteria
-   `CaptureTheFlagEnv` launches two SC2 clients successfully.
-   Two Pylons spawn at correct distances with reduced health.
-   Units spawn randomly based on resource limits.
-   Both agents are controlled by the same model policy.
-   Observation shape is confirmed as `(7, 128, 128)`.
-   Training pipeline runs for at least 1000 steps without crashing.
-   When stepping, the `command_assignment` layer highlights the active unit, and `command_objective` highlights the enemy Pylon.

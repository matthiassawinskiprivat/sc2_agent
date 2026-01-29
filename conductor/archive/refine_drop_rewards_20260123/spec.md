# Specification: Drop Level Reward Refinement

## Overview
This track aims to optimize the behavior of agents in the "Drop" level by refining the reward signals for loading and unloading units. The current behavior involves excessive loading/unloading; these changes incentivize efficient transport operations, such as filling transports before moving and rewarding "emergency" pickups for damaged units.

## Functional Requirements

### 1. Refined Loading Rewards
*   **Initial Load Reward (One-time per unit):**
    *   The agent receives a reward for loading an allied ground unit into a transport for the first time in an episode.
    *   **Scaling:** The reward scales based on the current cargo count of the transport *before* the unit is loaded.
    *   **Formula:** `Reward = Base_Load_Reward * (1 + (Current_Cargo_Count * 0.25))`
    *   *Goal:* Encourage gathering units into a single transport before departing.
*   **Emergency Pick-up Reward (Recurring):**
    *   If a unit has already been loaded once (Initial Load is False), the agent can still receive a reward for loading it if the unit's health is below 50% (Health + Shields for Protoss).
    *   **Scaling:** This reward is a flat `Base_Load_Reward` and does not scale with cargo count.
    *   *Goal:* Incentivize "saving" damaged units without rewarding infinite load/unload loops of healthy units.

### 2. Refined Unloading Rewards
*   **Target Zone Unload (One-time per unit):**
    *   Unloading a unit in the designated target zone is rewarded only once per unit per episode.
*   **Efficiency Scaling:**
    *   The reward for unloading in the target zone scales based on how full the transport was *before* the specific unload action started.
    *   **Formula:** `Reward = Base_Unload_Reward * (1 + ((Cargo_Before_Unload - 1) * 0.25))`
    *   *Example (Capacity 4):* 
        *   1st unit unloaded (Transport had 4): `1.75 * Base`
        *   2nd unit unloaded (Transport had 3): `1.50 * Base`
        *   ...
        *   Last unit unloaded (Transport had 1): `1.0 * Base`
    *   *Goal:* Encourage full-load drops rather than single-unit shuttle runs.

## Technical Constraints & Implementation Details
*   **Tracking:** Must maintain an episode-level registry of unit tags that have been "Initially Loaded" and "Target Unloaded".
*   **Observation:** Requires access to the `cargo_slots_used` or equivalent property of transport units.
*   **Health Calculation:** For Protoss, health must be calculated as `health + shield`.

## Acceptance Criteria
1.  Loading a healthy unit for the first time gives a reward that increases with transport fullness.
2.  Loading a healthy unit for the second time gives zero reward.
3.  Loading a damaged unit (<50% HP) for the second time gives a flat reward.
4.  Unloading a unit in the target zone for the first time gives a reward that is highest for the first unit of a full transport drop.
5.  Unloading the same unit in the target zone a second time gives zero reward.

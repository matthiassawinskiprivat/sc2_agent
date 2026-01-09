# Specification: CTF Distance Reward Integration

## Overview
This track implements a distance-based reward system for the Capture the Flag (CTF) environment, similar to the logic found in the Level 0 (Collect the Beacon) environment. The goal is to provide a dense reward signal that encourages units to move toward the enemy pylon and penalizes movement away from it.

## Functional Requirements
- **Unit-Specific Distance Tracking:** The reward system must track the distance of individual units to the target (enemy pylon) to ensure that rewards are calculated correctly in a multi-unit environment.
- **Sequential Reward Calculation:** Only the unit currently being stepped (the "active" unit) should receive/trigger the distance reward in a given step.
- **Configurable Weight:** The distance reward must be scalable via a `distance_multiplier` parameter in the environment configuration (`.yaml` files).
- **Positive/Negative Signal:** 
    - Progress toward the enemy pylon yields a positive reward.
    - Movement away from the enemy pylon yields a negative reward (penalty).
- **Additive Integration:** The distance reward must be added to existing rewards (pylon damage, combat rewards, etc.) rather than replacing them.

## Technical Requirements
- **Location:** Implementation should be integrated into `CaptureTheFlagBot.calculate_reward` within `src/agent/ctf_bot.py`.
- **State Persistence:** Use a dictionary to store previous distances per unit tag to prevent cross-unit reward contamination.
- **Configuration:** Update `config/capture_the_flag_0.yaml` (and other relevant CTF configs) to include `distance_multiplier` in the `rewards` section.

## Acceptance Criteria
1.  **Test Case - Forward Progress:** A unit moving closer to the enemy pylon receives a reward proportional to `dist_change * distance_multiplier`.
2.  **Test Case - Retreating:** A unit moving away from the enemy pylon receives a penalty.
3.  **Test Case - Multiple Units:** In a scenario with multiple units, moving Unit A does not affect the distance baseline for Unit B.
4.  **Test Case - Configurable:** Changing `distance_multiplier` in the yaml file reflects in the magnitude of the calculated rewards.
5.  **Integration:** Existing tests for CTF must pass to ensure no regressions in combat or termination logic.

# Specification - Fix CTF Distance and HP Rewards

This track addresses bugs in the Capture the Flag (CTF) environment's reward system that cause the agent to idle in the middle of the map. The issues stem from the Fog of War (FoW) handling, where hidden Pylons are treated as destroyed or non-existent for reward calculations.

## Core Issues

1.  **Distance Reward Skipping:** The distance reward logic `if active_unit and enemy_pylon:` fails when the enemy Pylon is in the Fog of War (`enemy_pylon` is None). This removes the guidance signal needed for the agent to navigate towards the objective.
2.  **False Positive Destruction:** The HP tracking logic incorrectly interprets a missing Pylon (due to FoW) as having 0 HP. This triggers a massive "Pylon Destroyed" reward, causing the agent to believe the objective is complete and subsequently idle.

## Requirements

*   **Persistent Targeting:** Use the persistent `self.enemy_pylon_pos` (last known position) for distance reward calculations instead of relying on immediate visibility.
*   **Fog of War Awareness:** Modify the HP reward calculation to check `self.is_visible(target_pos)`.
    *   If the Pylon unit is missing but the position is **visible**, it is confirmed dead (0 HP).
    *   If the Pylon unit is missing and the position is **hidden**, it is assumed alive (HP unchanged).
*   **No False Positives:** Ensure that moving out of range of the Pylon does not trigger a destruction reward.
*   **Consistent Guidance:** Ensure the distance reward effectively guides the agent towards the Pylon even when it is not currently visible on screen.

## Acceptance Criteria

*   `tests/test_ctf_fog_of_war.py` (to be recreated) passes, confirming:
    *   Distance rewards are generated when the Pylon is hidden.
    *   No HP reward is generated when the Pylon simply enters Fog of War.
    *   Correct HP reward is generated when the Pylon is actually destroyed (position visible, unit missing).
*   Existing tests in `tests/test_ctf_logic.py` and `tests/test_ctf_objective_logic.py` continue to pass.

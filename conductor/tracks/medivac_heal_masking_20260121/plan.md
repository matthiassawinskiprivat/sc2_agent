# Plan: Medivac Heal Action Masking

## Strategy
1.  **Investigation:**
    *   Confirm the `HEAL_RANGE` for Medivacs (Standard is 4.0, same as Load).
    *   Verify the existing `action_masks` logic in `src/sc2_bot.py`.

2.  **Implementation:**
    *   **Step 1: Update `src/sc2_bot.py`.**
        *   In `action_masks()`, within the `attack` head logic:
        *   If `active_unit` is a Medivac:
            *   Find all `self.units.filter(lambda u: u.is_biological and u.health < u.health_max)`.
            *   If no such unit is within `HEAL_RANGE + r1 + r2`, use `zero_mask`.
            *   Else, use `move_mask`.

3.  **Verification:**
    *   Create a test case `tests/test_medivac_heal_mask.py`.
    *   Scenario A: Medivac near full-health Marine -> Attack mask should be OFF (zeros).
    *   Scenario B: Medivac near damaged Marine -> Attack mask should be ON (move_mask).
    *   Scenario C: Marine active -> Attack mask should be unaffected (move_mask).

## Todo List
- [ ] Implement `HEAL_RANGE` check in `src/sc2_bot.py`.
- [ ] Create `tests/test_medivac_heal_mask.py`.
- [ ] Verify fix and ensure no regressions for other units.

# Plan: Enhance Load Action Masking

## Strategy
1.  **Investigation:**
    *   Verify how `sc2_bot.py` or `unit_control_agent.py` currently handles masking.
    *   Identify where the mask is generated (`get_action_mask`, `get_transport_mask`).
    *   Confirm how to access `cast_range` effectively without repeated proto lookups (cache it).

2.  **Implementation:**
    *   **Step 1: Cache Cast Ranges.**
        *   Create a utility or class attribute to store `cast_range` for supported transport types to avoid per-frame lookups.
    *   **Step 2: Update `get_transport_mask` (or equivalent).**
        *   Add logic to check distances between active unit and potential targets.
        *   If `active_unit` is transport: Check distance to friendly ground units.
        *   If `active_unit` is ground: Check distance to friendly transports.
        *   If no unit is in "potential range" (cast_range + radii), return a fully zeroed mask for the load head.
        *   Otherwise, proceed with the existing mask generation.

3.  **Verification:**
    *   Create a test case with a transport and a ground unit far apart -> Expect empty load mask.
    *   Create a test case with a transport and a ground unit close -> Expect non-empty load mask.
    *   Verify for different unit types (Medivac, Dropperlord, Warp Prism).

## Todo List
- [ ] Locate the exact masking function in `src/agent/unit_control_agent.py` or `src/sc2_bot.py`.
- [ ] Create a reproduction/test script `tests/test_load_mask_range.py` to verify current behavior (should likely be unmasked even if far).
- [ ] Implement `get_static_cast_range` helper.
- [ ] Modify masking logic in `src/agent/unit_control_agent.py` to implement the distance check.
- [ ] Run `tests/test_load_mask_range.py` to verify fix.
- [ ] Run existing tests to ensure no regressions.

# Track Spec: Fix Hidden Drop Reward Bugs

## Goal
Resolve critical logic errors in the reward system that prevent the agent from learning the drop sequence and correctly valuing objective destruction.

## Requirements
1. **Generic Combat Reward Logic:**
    - `calculate_combat_reward` must be updated to track damage and kills for **all** enemy units and structures, not just a specific `enemy_unit_type`.
    - This ensures damage to Command Centers or other objective structures yields rewards.

2. **Fix Distance Reward "Theft":**
    - `previous_transport_distances` and `previous_units_in_zone` should only be updated at the **end of a full unit cycle**, or the `dist_diff` calculation must be decoupled from the update to ensure rewards are not stolen by prior unit steps in the same frame.

3. **Fix Allied Attrition Penalty Multiplier:**
    - `calculate_allied_attrition_penalty` must ensure that a single unit death is only penalized **once per frame/cycle**. 
    - The deletion of dead tags from tracking dictionaries must be synchronized with the penalty application.

4. **Reward Consistency:**
    - Ensure `active_tag` attribution is strictly enforced so that only the unit performing the progress (or the transport carrying it) gets the shaping reward.

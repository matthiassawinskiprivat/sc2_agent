# Track Spec: Optimize Drop Reward Logic

## Goal
Refine the reward logic for the Drop environment to resolve conflicting incentives and ensure robust learning across all races and transport types.

## Requirements
1. **Weighted Distance Reward for Transports:**
    - Transports should receive a distance-shaping reward scaled by the number of units they are currently carrying.
    - `reward = (prev_dist - curr_dist) * scale * cargo_used`.
    - This ensures moving toward the objective is more rewarding when loaded.

2. **Fix "Fake Death" Loading Penalty:**
    - Loading a unit into a transport should not trigger an allied attrition penalty (kill penalty).
    - The attrition logic must account for units that are currently passengers.

3. **Exclude Non-Transport Units from Distance Reward:**
    - Ground units (infantry) in the drop environment should not receive distance-shaping rewards toward the enemy island.
    - This prevents them from "stalling" at the shore and focuses the objective-reaching task on the transports.

4. **Multi-Race / Multi-Transport Support:**
    - Logic must be generic enough to handle Medivacs (Terran), Overlords (Zerg), and Warp Prisms (Protoss).

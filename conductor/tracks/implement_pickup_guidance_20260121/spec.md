# Track Spec: Implement Pickup Guidance

## Goal
Provide shaping rewards to guide empty transports and infantry units toward each other, enabling the agent to discover the loading sequence.

## Requirements
1. **Empty Transport Guidance:**
    - Transports with `cargo_used == 0` should receive a distance-shaping reward for moving closer to the nearest allied infantry unit.
    - This signal should disappear or be replaced by the objective signal as soon as the transport is loaded.

2. **Infantry Guidance:**
    - Infantry units should receive a distance-shaping reward for moving closer to the nearest allied transport.
    - This encourages proactive "meeting" behavior.

3. **Smooth Transition:**
    - The logic must ensure that as units get closer, the rewards don't "fight" each other (e.g., oscillating between targets).
    - Distance tracking must be synchronized with the cycle-end updates.

4. **Multi-Race Support:**
    - Must work for all transport types (Medivac, Overlord, Warp Prism) and infantry types.

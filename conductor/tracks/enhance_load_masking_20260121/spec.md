# Track: Enhance Load Action Masking

## Context
The agent currently spends exploration time trying to load units when no valid targets are nearby. This creates noise and slows down learning in sparse-reward scenarios like drop operations. By masking the load action head when loading is physically impossible, we can guide the agent towards valid interactions.

## Goals
-   Mask the entire 'load' action head when the active transport unit is out of range of any loadable ground unit.
-   Mask the entire 'load' action head when the active ground unit is out of range of any transport unit.
-   Support multiple transport types (Medivac, Dropperlord, Warp Prism).
-   Maintain existing spatial masking logic when the action is generally valid (i.e., if in range, only mask invalid specific targets).

## Requirements
1.  **Transport Logic:**
    *   If the active agent is a transport:
        *   Calculate distance to all friendly ground units.
        *   If `min(distance) > cast_range + r_transport + r_target`: Mask ALL load actions.
        *   Otherwise: Use existing masking logic.
2.  **Ground Unit Logic:**
    *   If the active agent is a ground unit:
        *   Calculate distance to all friendly transport units.
        *   If `min(distance) > cast_range + r_transport + r_target`: Mask ALL load actions.
        *   Otherwise: Use existing masking logic.
3.  **Data Source:**
    *   Use `AbilityId` proto data to get static `cast_range` for relevant load abilities.
    *   Use unit `radius` properties.
4.  **Performance:**
    *   Use numpy for vectorized distance calculations to minimize overhead.

## Non-Goals
-   Changing the reward function.
-   Changing the observation space.

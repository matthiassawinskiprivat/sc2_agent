# Plan: Improve Action Masking

## Goal
Optimize action masking for Attack, Load, and Unload heads to improve learning efficiency and robustness across different races and environments.

## Tasks

- [x] **1. Attack Masking**: Update `src/sc2_bot.py` to use `move_mask` (or equivalent terrain mask) for the `attack` head instead of `unrestricted_mask` for ground units.
- [x] **2. Unload Masking**: Update `src/sc2_bot.py` to check `cargo_used`. If 0, return zero-mask (all False). If >0, use `get_unload_mask`. Also mask for non-transports.
- [x] **3. Load Masking (Transport Actor)**: Update `get_transport_mask` in `src/agent/unit_control_agent.py` to return `terrain_mask` (cliffs masked) when the actor is a transport.
- [x] **4. Load Masking (Ground Actor)**: Update `get_transport_mask` in `src/agent/unit_control_agent.py` to draw the target transport using its physical radius (circle) instead of a single point.
- [x] **5. Flying Unit Safety**: Ensure flying units (non-transports) have load/unload completely masked.

## Constraints
- Do not break existing functionality for other levels.
- Maintain compatibility with 128x128 grid size.
- Ensure efficient calculation (avoid heavy loops in critical path).

## Detailed Logic

### Attack Mask
- For ground units: Use `get_action_mask()` (Pathing grid).
- For flying units: Keep unrestricted (or mask only off-map).

### Load Mask
- **If Actor is Transport (e.g. Medivac):**
  - Return `get_action_mask()` (Terrain mask).
  - Rationale: Medivac should look for units on the ground. Masking cliffs helps efficiency.
- **If Actor is Ground (e.g. Marine):**
  - Iterate through allied transports.
  - Draw a filled circle (radius = unit radius + margin?) on the mask for each transport.
  - Rationale: Increase probability mass for clicking the transport.
- **If Actor is Non-Transport Flyer (e.g. Viking):**
  - Mask ALL (False).

### Unload Mask
- **If Actor is Transport:**
  - If `cargo_used > 0`: Use `get_unload_mask()`.
  - If `cargo_used == 0`: Mask ALL.
- **If Actor is Ground or Non-Transport:**
  - Mask ALL.
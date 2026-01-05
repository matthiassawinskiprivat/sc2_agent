# Track Specification: Perspective Flipping for Self-Play

## 1. Overview
Implement robust "Perspective Flipping" for the Capture the Flag (CTF) environment. This ensures that regardless of which player (P1 or P2) the agent is controlling, the observation and action spaces are transformed to a canonical "Player 1" perspective (e.g., always starting on the left). This allows a single shared model to effectively learn and generalize for both sides of the map in self-play scenarios.

## 2. Goals
- **Symmetry**: Ensure the agent receives visually identical inputs for symmetric game states, regardless of player ID.
- **Reusability**: Implement the core flipping logic as a standalone utility or wrapper that can be applied to future environments (e.g., Level 7 Combat, future symmetrical maps).
- **Correctness**: Verify that not just images, but also actions (movement commands) and potentially reward signals are correctly inverted.

## 3. Scope
### In Scope
- **Analysis**: Investigate current observation generation (`ObservationTransformer`) and action processing (`UnitControlAgent`) to identify all coordinate-dependent logic.
- **Implementation**:
    - Create a reusable `PerspectiveTransformer` (or similar utility) that handles:
        - Flipping 2D feature maps (Observations).
        - Inverting spatial actions (Move/Attack coordinates).
    - Integrate this transformer into `CaptureTheFlagEnv` (or a specific wrapper).
- **Testing**:
    - Unit tests verifying that a P2 observation is the exact geometric flip of the equivalent P1 observation.
    - Unit tests verifying that a "Move Right" action from the agent results in a "Move Left" execution for P2.

### Out of Scope
- Changing the underlying map generation (random flags remain).
- Modifying the model architecture (CNN).

## 4. Key Questions to Resolve (during Investigation)
- **Reward Calculation**: Does `CaptureTheFlagBot.calculate_reward` already handle perspective (e.g., "distance to *enemy* base") correctly, or does it also rely on absolute coordinates that need flipping?
- **Masking**: Does `action_masks()` need flipping? (e.g., if the map has asymmetric walls).

## 5. Acceptance Criteria
1. **Visual Symmetry**: If P1 is at (10, 10) and P2 is at (118, 118) on a 128x128 grid, P1's view of itself must be identical to P2's view of itself (after flipping).
2. **Action Consistency**: An agent outputting "Action X" should result in symmetric behavior for both players (e.g., both advancing toward the center).
3. **Tests Pass**: New unit tests for the transformer/wrapper pass.

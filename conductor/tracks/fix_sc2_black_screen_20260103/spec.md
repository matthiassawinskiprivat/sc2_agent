# Specification: Fix SC2 Black Screen

## 1. Overview
When running SC2 with visualization enabled (`visualize=True`), the game window sometimes displays only a black screen. Additionally, setting `fullscreen=False` (to run in windowed mode) may unexpectedly result in fullscreen mode. This track aims to ensure the game renders correctly and respects window/fullscreen settings.

## 2. Functional Requirements
- **Rendering Fix:** Ensure the SC2 process renders the game world correctly in the window.
- **Window Management:** Correct the behavior of the `fullscreen` parameter to ensure windowed mode is used when requested.
- **Visualization Support:** Allow users to watch the game in non-headless mode without visual glitches.

## 3. Acceptance Criteria
- SC2 game windows show the game world (not a black screen) when `visualize=True`.
- Setting `visualize=True` (which passes `fullscreen` to `SC2Process`) results in a windowed or fullscreen mode that displays the game.
- If `visualize=False`, the game remains headless/invisible as expected.

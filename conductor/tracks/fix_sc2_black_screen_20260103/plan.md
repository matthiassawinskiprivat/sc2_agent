# Plan: Fix SC2 Black Screen

This plan outlines the steps to investigate and fix rendering issues and window mode bugs.

## Phase 1: Investigation & Fix [checkpoint: current]
Investigate `SC2Process` launch parameters and rendering state.

- [x] Task: Inspect `MultiClientSC2Manager.launch_processes` and how it uses `fullscreen`.
- [x] Task: Check if `realtime` or `rgb_render_config` affects the black screen. (Added `render=True` when visualizing).
- [x] Task: Research `burnysc2` `SC2Process` arguments for windowed vs fullscreen.
- [x] Task: Implement fix for black screen and fullscreen bug by separating `visualize` and `fullscreen` flags and enabling `render`.
- [ ] Task: Verify with manual test or user feedback.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Investigation & Fix' (Protocol in workflow.md)
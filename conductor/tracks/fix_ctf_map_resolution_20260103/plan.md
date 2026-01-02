# Plan: Fix CTF Map Resolution and Attributes

This plan outlines the steps to resolve map discovery and initialization issues in `MultiClientEnv`.

## Phase 1: Implementation & Verification [checkpoint: 7da418b]
Fix the base class and verify map resolution.

- [x] Task: Implement map resolution and discovery in `MultiClientEnv`.
- [x] Task: Update `CaptureTheFlagEnv` to use the new map resolution.
- [x] Task: Fix `AttributeError` by moving space placeholders to `__init__`.
- [x] Task: Create a regression test `tests/test_multi_client_map_resolve.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Implementation & Verification' (Protocol in workflow.md)
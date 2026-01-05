# Implementation Plan - Air Unit Unrestricted Movement

## Phase 1: Environment & Bot Logic
- [x] Task: Create a reproduction test case for air unit masking `tests/test_air_unit_masking.py` 057a58d
- [x] Task: Update `CTFBot.action_masks` to bypass masking for air units 057a58d
- [x] Task: Update `Level7Bot.action_masks` to bypass masking for air units 057a58d
- [x] Task: (Optional) Refactor `UnitControlAgent` to provide a helper for "full" masks (Handled in bots)

## Phase 2: Verification
- [x] Task: Run `tests/test_air_unit_masking.py` and verify all tests pass 057a58d
- [x] Task: Run existing level 7 and CTF tests to ensure no regressions for ground units 057a58d
- [x] Task: Update `product.md` documentation regarding air unit mechanics 057a58d

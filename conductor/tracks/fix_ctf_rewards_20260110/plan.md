# Plan - Fix CTF Distance and HP Rewards

This plan addresses the idling behavior in CTF agents caused by incorrect reward handling under Fog of War conditions.

## Todo List

- [x] Task: Create reproduction test suite `tests/test_ctf_fog_of_war.py` that asserts failure on current codebase (idling/false rewards). [checkpoint: 9ecb4b4]
- [x] Task: Update `src/agent/ctf_bot.py` to use `self.enemy_pylon_pos` for distance calculations. [checkpoint: 9ecb4b4]
- [x] Task: Update `src/agent/ctf_bot.py` to implement `get_enemy_hp_safe` logic that checks `is_visible` before assuming Pylon death. [checkpoint: 9ecb4b4]
- [x] Task: Verify fix with `tests/test_ctf_fog_of_war.py` and ensure all tests pass. [checkpoint: 9ecb4b4]

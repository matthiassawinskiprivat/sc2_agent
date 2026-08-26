"""Opponent seating config for manager / multi-agent curricula."""

from __future__ import annotations

from typing import Any, Dict


def resolve_opponent_config(training_conf: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse ``env.opponent`` (or flattened ``opponent``) for Computer vs unit-control.

    Shape (forward-compatible)::

        opponent:
          mode: computer | unit_control
          difficulty: Easy          # computer only
          agent: zerg_unit_control  # unit_control only (later)
          checkpoint: path/to.zip   # unit_control only (later)
          freeze: true              # unit_control only (later)
    """
    env = training_conf.get("env", {})
    raw = None
    if isinstance(env, dict) and isinstance(env.get("opponent"), dict):
        raw = env["opponent"]
    elif isinstance(training_conf.get("opponent"), dict):
        raw = training_conf["opponent"]

    difficulty = "Easy"
    if isinstance(env, dict) and env.get("difficulty"):
        difficulty = str(env["difficulty"])
    elif training_conf.get("difficulty"):
        difficulty = str(training_conf["difficulty"])

    if raw is None:
        return {
            "mode": "computer",
            "difficulty": difficulty,
            "agent": None,
            "checkpoint": None,
            "freeze": True,
        }

    mode = str(raw.get("mode", "computer")).lower()
    return {
        "mode": mode,
        "difficulty": str(raw.get("difficulty", difficulty)),
        "agent": raw.get("agent"),
        "checkpoint": raw.get("checkpoint"),
        "freeze": bool(raw.get("freeze", True)),
    }

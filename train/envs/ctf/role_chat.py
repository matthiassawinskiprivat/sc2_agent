"""In-game chat lines for CTF trainee vs opponent role (post-debug spawn)."""
from __future__ import annotations

from typing import List, Optional, Sequence


def _fmt_race(race: Optional[str]) -> str:
    text = str(race or "?").strip() or "?"
    return text[:1].upper() + text[1:].lower()


def _fmt_roles(roles: Sequence[str]) -> str:
    parts = [str(r).strip() for r in roles if str(r).strip()]
    return ",".join(parts) if parts else "?"


def format_ctf_role_chat_lines(
    *,
    episode: int,
    trainee_race: Optional[str],
    trainee_player_id: int,
    trainee_roles: Sequence[str],
    opponent_race: Optional[str],
    opponent_player_id: int,
    opponent_roles: Sequence[str],
    opponent_policy: str,
) -> List[str]:
    """Two short SC2 chat lines: trainee policy, then opponent policy."""
    ep = max(int(episode), 1)
    opp = str(opponent_policy or "none").strip() or "none"
    return [
        (
            f"Ep{ep} TRAINEE: {_fmt_race(trainee_race)} {_fmt_roles(trainee_roles)} "
            f"(P{int(trainee_player_id)})"
        ),
        (
            f"Ep{ep} OPPONENT: {_fmt_race(opponent_race)} {_fmt_roles(opponent_roles)} "
            f"(P{int(opponent_player_id)}, {opp})"
        ),
    ]

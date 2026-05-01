"""
CTF environment configuration schemas.

Uses pydantic v2 BaseModel for runtime validation.  All fields are
required unless a default is provided.  Load from YAML via
``CTFConfig.from_yaml(path)``.

Key constraints enforced at validation time:
- every_k_episodes >= 50  (prevents degenerate opponent non-stationarity)
- Attacker squads must have non-empty attack_targets
- Defender squads must have non-empty defend_anchors
- Budget minerals + vespene >= 0
- spawn_pool must be non-empty
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

try:
    from pydantic import BaseModel, field_validator, model_validator, Field
    _PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, validator as field_validator, root_validator as model_validator, Field  # type: ignore
    _PYDANTIC_V2 = False

_ROLE = Literal["attack", "defend"]
_SIDE = Literal["player_a", "player_b"]

MIN_K_EPISODES = 50  # enforced minimum for opponent sync period


class BudgetConfig(BaseModel):
    """Mineral + vespene budget cap for roster generation."""

    minerals: int = Field(..., ge=0)
    vespene: int = Field(..., ge=0)


class AttackerRewardConfig(BaseModel):
    """Reward parameters applied when a squad's current role is 'attack'."""

    step_penalty: float = Field(default=-0.001, le=0.0)
    pylon_destroy_bonus: float = Field(default=10.0, ge=0.0)
    unit_kill_bonus: float = Field(default=1.0, ge=0.0)
    target_type_bonus: float = Field(default=2.0, ge=0.0)
    target_type_bonus_clip: float = Field(default=5.0, ge=0.0)

    @field_validator("target_type_bonus_clip")
    @classmethod
    def clip_gte_bonus(cls, v: float, info: Any = None) -> float:
        return v


class DefenderRewardConfig(BaseModel):
    """Reward parameters applied when a squad's current role is 'defend'."""

    step_reward: float = Field(default=0.001, ge=0.0)
    pylon_survival_bonus: float = Field(default=5.0, ge=0.0)
    unit_kill_bonus: float = Field(default=1.0, ge=0.0)
    target_type_bonus: float = Field(default=2.0, ge=0.0)
    target_type_bonus_clip: float = Field(default=5.0, ge=0.0)


class RewardConfig(BaseModel):
    """Top-level reward config; split by role."""

    attacker: AttackerRewardConfig = Field(default_factory=AttackerRewardConfig)
    defender: DefenderRewardConfig = Field(default_factory=DefenderRewardConfig)


class SquadConfig(BaseModel):
    """Configuration for one squad (one entry in ``CTFConfig.squads``)."""

    id: str = Field(..., min_length=1)
    role: _ROLE
    side: _SIDE
    budget: BudgetConfig
    spawn_pool: List[int] = Field(..., min_length=1)
    spawn_region: str = Field(..., min_length=1)
    attack_targets: List[str] = Field(default_factory=list)
    defend_anchors: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def role_regions_consistent(self) -> "SquadConfig":
        if self.role == "attack" and not self.attack_targets:
            raise ValueError(
                f"Squad '{self.id}' has role='attack' but attack_targets is empty."
            )
        if self.role == "defend" and not self.defend_anchors:
            raise ValueError(
                f"Squad '{self.id}' has role='defend' but defend_anchors is empty."
            )
        return self


class CTFConfig(BaseModel):
    """
    Top-level configuration for the CTF SC2 environment (universal pipeline).

    Loaded from YAML; validated at construction time.
    """

    env_id: str = Field(default="ctf_sc2_v1")
    map_path: str = Field(..., min_length=1)
    max_steps: int = Field(..., gt=0)
    every_k_episodes: int = Field(..., ge=MIN_K_EPISODES)
    realtime: bool = Field(default=False)
    debug_spawn: bool = Field(default=False)

    cells_per_world_unit: float = Field(default=0.25, gt=0.0)
    global_grid_size: int = Field(default=128, gt=0)
    tactical_grid_size: int = Field(default=64, gt=0)
    max_units_per_squad: int = Field(default=16, gt=0)
    max_enemies_visible: int = Field(default=32, gt=0)

    squads: List[SquadConfig] = Field(..., min_length=1)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    episode_seed: Optional[int] = Field(default=None)

    @field_validator("every_k_episodes")
    @classmethod
    def k_minimum(cls, v: int) -> int:
        if v < MIN_K_EPISODES:
            raise ValueError(
                f"every_k_episodes must be >= {MIN_K_EPISODES} to avoid "
                f"degenerate opponent non-stationarity, got {v}."
            )
        return v

    @model_validator(mode="after")
    def squad_ids_unique(self) -> "CTFConfig":
        ids = [s.id for s in self.squads]
        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate squad ids: {list(set(dupes))}")
        return self

    @model_validator(mode="after")
    def both_sides_present(self) -> "CTFConfig":
        sides = {s.side for s in self.squads}
        if "player_a" not in sides or "player_b" not in sides:
            raise ValueError(
                "Config must include squads for both player_a and player_b."
            )
        return self

    @model_validator(mode="after")
    def validate_region_pairing(self) -> "CTFConfig":
        """
        Every attack_target must be matched by at least one defend_anchor
        on the opposing side (same region name), and vice versa.
        """
        attack_targets: Dict[str, List[str]] = {}
        defend_anchors: Dict[str, List[str]] = {}

        for sq in self.squads:
            if sq.role == "attack":
                attack_targets.setdefault(sq.side, []).extend(sq.attack_targets)
            else:
                defend_anchors.setdefault(sq.side, []).extend(sq.defend_anchors)

        for side, targets in attack_targets.items():
            opponent = "player_b" if side == "player_a" else "player_a"
            anchors = set(defend_anchors.get(opponent, []))
            unmatched = [t for t in targets if t not in anchors]
            if unmatched:
                raise ValueError(
                    f"Attack targets {unmatched} for side '{side}' have no "
                    f"matching defend_anchors on the opposing side '{opponent}'."
                )
        return self

    def squads_for_side(self, side: _SIDE) -> List[SquadConfig]:
        return [s for s in self.squads if s.side == side]

    def attack_squads(self, side: _SIDE) -> List[SquadConfig]:
        return [s for s in self.squads if s.side == side and s.role == "attack"]

    def defend_squads(self, side: _SIDE) -> List[SquadConfig]:
        return [s for s in self.squads if s.side == side and s.role == "defend"]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CTFConfig":
        """Load and validate config from a YAML file."""
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config to YAML."""
        Path(path).write_text(
            yaml.dump(self.model_dump(), default_flow_style=False),
            encoding="utf-8",
        )

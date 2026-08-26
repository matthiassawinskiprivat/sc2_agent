"""Per-unit scalar feature index mapping for the squad pointer key.

Mirrors the LayerConstants pattern: named string constants, a FEATURES index dict,
and get_dim(). Append new features at the END of FEATURES — never reorder existing
entries, so trained models stay compatible with the index they learned.
"""


class UnitScalarConstants:
    # === Group A: Vitals ===
    HEALTH_PCT           = "health_pct"
    SHIELD_PCT           = "shield_pct"
    COMBINED_VITALITY    = "combined_vitality"
    WEAPON_COOLDOWN_NORM = "weapon_cooldown_norm"
    WEAPON_READY         = "weapon_ready"

    # === Group B: Kiting / mobility ===
    MOVEMENT_SPEED_NORM       = "movement_speed_norm"
    DIST_TO_WEAPON_READY_NORM = "dist_to_weapon_ready_norm"
    ENERGY_PCT                = "energy_pct"

    # === Group C: Static combat profile (per unit type, upgrade-aware) ===
    GROUND_RANGE_NORM = "ground_range_norm"
    AIR_RANGE_NORM    = "air_range_norm"
    GROUND_DPS_NORM   = "ground_dps_norm"
    CAN_ATTACK_AIR    = "can_attack_air"
    IS_MELEE          = "is_melee"
    MAX_HEALTH_NORM   = "max_health_norm"

    # === Group D: Unit type attribute flags ===
    IS_BIOLOGICAL = "is_biological"
    IS_MECHANICAL = "is_mechanical"
    IS_ARMORED    = "is_armored"
    IS_MASSIVE    = "is_massive"
    IS_FLYING     = "is_flying"
    HAS_SHIELD    = "has_shield"

    # === Group E: Positional / relational ===
    X_NORM                  = "x_norm"
    Y_NORM                  = "y_norm"
    DIST_TO_OBJECTIVE_NORM  = "dist_to_objective_norm"
    DIST_TO_ENEMY_COM_NORM  = "dist_to_enemy_com_norm"
    DIST_TO_ALLIED_COM_NORM = "dist_to_allied_com_norm"

    # === Group F: Engagement / concealment state ===
    IS_ATTACKING = "is_attacking"
    IS_MOVING    = "is_moving"
    IS_CONCEALED = "is_concealed"
    IS_DETECTED  = "is_detected"

    # === Group G: Pool role ===
    ROLE_ATTACK = "role_attack"
    ROLE_DEFEND = "role_defend"

    # === Group H: Combat profile extensions (append-only) ===
    AIR_DPS_NORM     = "air_dps_norm"
    HAS_SPLASH       = "has_splash"
    ONETIME_UNIT     = "onetime_unit"
    IS_FRIENDLY_FIRE = "is_friendly_fire"
    IS_TEMPORARY     = "is_temporary"

    # === Group I: Cloak / immobility (append-only) ===
    IS_IMMOBILE = "is_immobile"
    CAN_CLOAK   = "can_cloak"

    # === Group J: Heal / repair readiness (append-only) ===
    CAN_HEAL   = "can_heal"
    CAN_REPAIR = "can_repair"

    # === Group K: Transport / cargo (append-only) ===
    IS_TRANSPORT     = "is_transport"
    CARGO_USED_NORM  = "cargo_used_norm"
    CARGO_FREE_NORM  = "cargo_free_norm"
    CARGO_SIZE_NORM  = "cargo_size_norm"
    CAN_LOAD         = "can_load"
    CAN_UNLOAD       = "can_unload"

    # === Group L: AoE spell readiness (append-only) ===
    CAN_AOE = "can_aoe"

    # --- INDEX MAPPING — never reorder; append new entries at the end ---
    FEATURES = {
        # Group A
        HEALTH_PCT:           0,
        SHIELD_PCT:           1,
        COMBINED_VITALITY:    2,
        WEAPON_COOLDOWN_NORM: 3,
        WEAPON_READY:         4,
        # Group B
        MOVEMENT_SPEED_NORM:       5,
        DIST_TO_WEAPON_READY_NORM: 6,
        ENERGY_PCT:                7,
        # Group C
        GROUND_RANGE_NORM: 8,
        AIR_RANGE_NORM:    9,
        GROUND_DPS_NORM:   10,
        CAN_ATTACK_AIR:    11,
        IS_MELEE:          12,
        MAX_HEALTH_NORM:   13,
        # Group D
        IS_BIOLOGICAL: 14,
        IS_MECHANICAL: 15,
        IS_ARMORED:    16,
        IS_MASSIVE:    17,
        IS_FLYING:     18,
        HAS_SHIELD:    19,
        # Group E
        X_NORM:                  20,
        Y_NORM:                  21,
        DIST_TO_OBJECTIVE_NORM:  22,
        DIST_TO_ENEMY_COM_NORM:  23,
        DIST_TO_ALLIED_COM_NORM: 24,
        # Group F
        IS_ATTACKING: 25,
        IS_MOVING:    26,
        IS_CONCEALED: 27,
        IS_DETECTED:  28,
        # Group G
        ROLE_ATTACK: 29,
        ROLE_DEFEND: 30,
        # Group H
        AIR_DPS_NORM:     31,
        HAS_SPLASH:       32,
        ONETIME_UNIT:     33,
        IS_FRIENDLY_FIRE: 34,
        IS_TEMPORARY:     35,
        # Group I
        IS_IMMOBILE: 36,
        CAN_CLOAK:   37,
        # Group J
        CAN_HEAL:   38,
        CAN_REPAIR: 39,
        # Group K
        IS_TRANSPORT:    40,
        CARGO_USED_NORM: 41,
        CARGO_FREE_NORM: 42,
        CARGO_SIZE_NORM: 43,
        CAN_LOAD:        44,
        CAN_UNLOAD:      45,
        # Group L
        CAN_AOE: 46,
    }

    @classmethod
    def get_dim(cls) -> int:
        """Total number of features in the master catalog."""
        return max(cls.FEATURES.values()) + 1

    @classmethod
    def default_active_features(cls) -> list:
        """All catalog features in stable index order (fallback when config is empty)."""
        return [name for name, _ in sorted(cls.FEATURES.items(), key=lambda kv: kv[1])]

    @classmethod
    def resolve_active_features(cls, names) -> list:
        """
        Validate and normalize a config list of feature names.

        Empty / None → full catalog in index order.
        Unknown names raise ValueError. Duplicates are skipped (first wins).
        """
        if not names:
            return cls.default_active_features()
        out = []
        for raw in names:
            name = str(raw).lower()
            if name not in cls.FEATURES:
                raise ValueError(
                    f"Unknown unit scalar feature {raw!r}. "
                    f"Known: {sorted(cls.FEATURES)}"
                )
            if name not in out:
                out.append(name)
        return out

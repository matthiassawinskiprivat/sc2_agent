"""
Layer Constants for StarCraft 2 Observation Transformer
46-Layer Architecture: High-Fidelity State Space with Orthogonal Accumulation

DOCUMENTATION LEGEND:
- Value Range: All layers are normalized.
  - Count Layers (0-13): log1p(count)
  - Vital Layers (14-17): Accumulation / Total Max
  - DPS/HPS Layers: tanh(value / 50.0)
- Strategy: Provides distinct channels for unit attributes to handle "pixel collisions" (stacking units).
"""

class LayerConstants:
    # ==============================================================================
    # GROUP 1: ALLIED UNIT STATE (Layers 0-6)
    # Describes the physical properties and readiness of our own units.
    # ==============================================================================

    # Layer 0: Density (Weighted by Readiness)
    # Logic: Accumulates (0.3 + 0.7*Readiness) per unit.
    # Strategy: High value = Many units ready to fight.
    ALLIED_UNIT_DENSITY     = "allied_unit_density"

    # Layers 1-4: Orthogonal Attribute Counters
    # Logic: Accumulate 1.0 for each unit with this tag.
    # Strategy: Allows AI to distinguish "10 Marines" vs "10 Marines + 2 Tanks".
    ALLIED_UNIT_BIOLOGICAL  = "allied_unit_biological"
    ALLIED_UNIT_MECHANICAL  = "allied_unit_mechanical"
    ALLIED_UNIT_ARMORED     = "allied_unit_armored"
    ALLIED_UNIT_MASSIVE     = "allied_unit_massive"

    # Layer 5: Movement Type
    # Logic: Accumulate 1.0 for Flyers. (Colossus adds 0.5 or 1.0 depending on logic).
    ALLIED_UNIT_FLYING      = "allied_unit_flying"

    # Layer 6: Cloak State
    # Logic: Accumulate 1.0 for Active Cloak, 0.5 for Cloak Capable.
    ALLIED_UNIT_CLOAK       = "allied_unit_cloak"


    # ==============================================================================
    # GROUP 2: ENEMY UNIT STATE (Layers 7-13)
    # Mirror of Group 1 for the opponent.
    # ==============================================================================

    ENEMY_UNIT_DENSITY      = "enemy_unit_density"     # Always adds 1.0 per unit (unknown readiness)
    ENEMY_UNIT_BIOLOGICAL   = "enemy_unit_biological"
    ENEMY_UNIT_MECHANICAL   = "enemy_unit_mechanical"
    ENEMY_UNIT_ARMORED      = "enemy_unit_armored"
    ENEMY_UNIT_MASSIVE      = "enemy_unit_massive"
    ENEMY_UNIT_FLYING       = "enemy_unit_flying"
    ENEMY_UNIT_CLOAK        = "enemy_unit_cloak"


    # ==============================================================================
    # GROUP 3: VITALS (Layers 14-17)
    # Accumulation of raw stats.
    # ==============================================================================

    # Logic: Sum(CurrentHP) / UnitMaxHP.
    # Note: If 10 units are here, this can go > 1.0. Log-normalization handles this.
    ALLIED_HEALTH = "allied_health"
    ENEMY_HEALTH  = "enemy_health"
    ALLIED_ENERGY = "allied_energy"
    ENEMY_ENERGY  = "enemy_energy"


    # ==============================================================================
    # GROUP 4: ALLIED OFFENSE MAPS (Layers 18-25)
    # Heatmaps of damage WE are projecting.
    # ==============================================================================

    ALLIED_G_DPS_BASE    = "allied_g_dps_base"
    ALLIED_G_DPS_LIGHT   = "allied_g_dps_light"
    ALLIED_G_DPS_ARMOR   = "allied_g_dps_armor"
    ALLIED_G_DPS_MASSIVE = "allied_g_dps_massive"

    ALLIED_A_DPS_BASE    = "allied_a_dps_base"
    ALLIED_A_DPS_LIGHT   = "allied_a_dps_light"
    ALLIED_A_DPS_ARMOR   = "allied_a_dps_armor"
    ALLIED_A_DPS_MASSIVE = "allied_a_dps_massive"


    # ==============================================================================
    # GROUP 5: ENEMY THREAT MAPS (Layers 26-33)
    # Heatmaps of damage projected BY THE ENEMY.
    # ==============================================================================

    ENEMY_G_DPS_BASE     = "enemy_g_dps_base"
    ENEMY_G_DPS_LIGHT    = "enemy_g_dps_light"
    ENEMY_G_DPS_ARMOR    = "enemy_g_dps_armor"
    ENEMY_G_DPS_MASSIVE  = "enemy_g_dps_massive"

    ENEMY_A_DPS_BASE     = "enemy_a_dps_base"
    ENEMY_A_DPS_LIGHT    = "enemy_a_dps_light"
    ENEMY_A_DPS_ARMOR    = "enemy_a_dps_armor"
    ENEMY_A_DPS_MASSIVE  = "enemy_a_dps_massive"


    # ==============================================================================
    # GROUP 6: UTILITY & SUSTAIN (Layers 34-35)
    # ==============================================================================

    ALLIED_HPS_FIELD     = "allied_hps_field"
    ENEMY_HPS_FIELD      = "enemy_hps_field"


    # ==============================================================================
    # GROUP 7: INTEL & ENVIRONMENT (Layers 36-39)
    # ==============================================================================

    ALLIED_DETECTION = "allied_detection"
    ENEMY_DETECTION  = "enemy_detection"
    TERRAIN_PATHABLE = "terrain_pathable"
    TERRAIN_CREEP    = "terrain_creep"


    # ==============================================================================
    # GROUP 8: PROTOCOL (Layers 40-41)
    # ==============================================================================

    COMMAND_ASSIGNMENT = "command_assignment"
    COMMAND_OBJECTIVE  = "command_objective"
    
    # ==============================================================================
    # GROUP 9: TRANSPORT (Layers 42-43)
    # ==============================================================================
    TRANSPORT_SLOTS    = "transport_slots"
    PASSENGER_DENSITY  = "passenger_density"

    # Layer 44: Allied weapon cooldown (tactical split from density; 0=ready, 1=max capped cd)
    ALLIED_WEAPON_COOLDOWN = "allied_weapon_cooldown"
    # Layer 45: Enemy weapon cooldown (tactical-only combat timing signal)
    ENEMY_WEAPON_COOLDOWN = "enemy_weapon_cooldown"

    # --- INDEX MAPPING ---
    LAYERS = {
        ALLIED_UNIT_DENSITY: 0,
        ALLIED_UNIT_BIOLOGICAL: 1, ALLIED_UNIT_MECHANICAL: 2,
        ALLIED_UNIT_ARMORED: 3, ALLIED_UNIT_MASSIVE: 4,
        ALLIED_UNIT_FLYING: 5, ALLIED_UNIT_CLOAK: 6,

        ENEMY_UNIT_DENSITY: 7,
        ENEMY_UNIT_BIOLOGICAL: 8, ENEMY_UNIT_MECHANICAL: 9,
        ENEMY_UNIT_ARMORED: 10, ENEMY_UNIT_MASSIVE: 11,
        ENEMY_UNIT_FLYING: 12, ENEMY_UNIT_CLOAK: 13,

        ALLIED_HEALTH: 14, ENEMY_HEALTH: 15,
        ALLIED_ENERGY: 16, ENEMY_ENERGY: 17,

        ALLIED_G_DPS_BASE: 18, ALLIED_G_DPS_LIGHT: 19, ALLIED_G_DPS_ARMOR: 20, ALLIED_G_DPS_MASSIVE: 21,
        ALLIED_A_DPS_BASE: 22, ALLIED_A_DPS_LIGHT: 23, ALLIED_A_DPS_ARMOR: 24, ALLIED_A_DPS_MASSIVE: 25,

        ENEMY_G_DPS_BASE: 26, ENEMY_G_DPS_LIGHT: 27, ENEMY_G_DPS_ARMOR: 28, ENEMY_G_DPS_MASSIVE: 29,
        ENEMY_A_DPS_BASE: 30, ENEMY_A_DPS_LIGHT: 31, ENEMY_A_DPS_ARMOR: 32, ENEMY_A_DPS_MASSIVE: 33,

        ALLIED_HPS_FIELD: 34, ENEMY_HPS_FIELD: 35,

        ALLIED_DETECTION: 36, ENEMY_DETECTION: 37,
        TERRAIN_PATHABLE: 38, TERRAIN_CREEP: 39,

        COMMAND_ASSIGNMENT: 40, COMMAND_OBJECTIVE: 41,
        
        TRANSPORT_SLOTS: 42, PASSENGER_DENSITY: 43,
        ALLIED_WEAPON_COOLDOWN: 44,
        ENEMY_WEAPON_COOLDOWN: 45,
    }

    # Reduced set for tactical/micro view
    TACTICAL_LAYERS = {
        ALLIED_UNIT_DENSITY: 0,
        ENEMY_UNIT_DENSITY: 1,
        TERRAIN_PATHABLE: 2,
        COMMAND_ASSIGNMENT: 3,
        COMMAND_OBJECTIVE: 4,
        ALLIED_G_DPS_BASE: 5,
        ENEMY_G_DPS_BASE: 6,
        ALLIED_HEALTH: 7,
        ENEMY_HEALTH: 8,
        TERRAIN_CREEP: 9,
        ALLIED_WEAPON_COOLDOWN: 10,
        ENEMY_WEAPON_COOLDOWN: 11,
        # Unit attribute flags (mirrors LAYERS group 1/2)
        ALLIED_UNIT_BIOLOGICAL: 12,
        ENEMY_UNIT_BIOLOGICAL: 13,
        ALLIED_UNIT_MECHANICAL: 14,
        ENEMY_UNIT_MECHANICAL: 15,
        ALLIED_UNIT_ARMORED: 16,
        ENEMY_UNIT_ARMORED: 17,
        ALLIED_UNIT_MASSIVE: 18,
        ENEMY_UNIT_MASSIVE: 19,
        # Typed DPS bonus channels (bonus damage per target attribute)
        ALLIED_G_DPS_LIGHT: 20,
        ENEMY_G_DPS_LIGHT: 21,
        ALLIED_G_DPS_ARMOR: 22,
        ENEMY_G_DPS_ARMOR: 23,
        ALLIED_G_DPS_MASSIVE: 24,
        ENEMY_G_DPS_MASSIVE: 25,
    }

    @classmethod
    def get_num_layers(cls) -> int:
        return 46

    @classmethod
    def get_num_tactical_layers(cls) -> int:
        return 26
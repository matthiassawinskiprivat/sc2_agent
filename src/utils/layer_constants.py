"""
Layer Constants for StarCraft 2 Observation Transformer
40-Layer Architecture: High-Fidelity State Space

DOCUMENTATION LEGEND:
- Value Range: All layers are normalized between [0.0, 1.0] unless specified.
- Logic: How the transformer calculates this value.
- Strategy: Why the Neural Network needs this specific information.
"""

class LayerConstants:
    # ==============================================================================
    # GROUP 1: ALLIED UNIT STATE (Layers 0-5)
    # Describes the physical properties and readiness of our own units.
    # ==============================================================================

    # Layer 0: Presence & Cooldown
    # Logic: 0.0=Empty, 0.5=Just Fired (Cooldown Active), 1.0=Ready to Fire
    # Strategy: Critical for Stutter-Stepping. AI learns to move when < 1.0 and shoot when == 1.0.
    ALLIED_UNIT_DENSITY = "allied_unit_density"

    # Layer 1: Armor Tag
    # Logic: 0.0=None, 0.5=Light, 1.0=Armored
    # Strategy: Self-awareness for defense. "I am Armored, I should avoid Immortal shots."
    ALLIED_UNIT_ARMOR   = "allied_unit_armor"

    # Layer 2: Biological vs Mechanical
    # Logic: 0.0=None, 0.5=Mechanical, 1.0=Biological, 0.75=Both (SCV)
    # Strategy: Determines healing eligibility (Medivac vs Repair) and vulnerability (Archons vs Bio).
    ALLIED_UNIT_TYPE    = "allied_unit_type"

    # Layer 3: Special Tags
    # Logic: 0.0=None, 0.5=Psionic, 1.0=Massive
    # Strategy: Vulnerability to specific counters (Ghost Snipe vs Psi, Tempest/Corruptor vs Massive).
    ALLIED_UNIT_SPECIAL = "allied_unit_special"

    # Layer 4: Movement Type
    # Logic: 0.0=Ground, 1.0=Flying, 0.5=Colossus (Tall/Hybrid)
    # Strategy: Navigation (can I cross cliffs?) and Targeting (can I be hit by AA?).
    ALLIED_UNIT_FLYING  = "allied_unit_flying"

    # Layer 5: Cloak State
    # Logic: 0.0=None, 0.3=Cloak Capable (Has energy/ability), 0.6=Active Cloak
    # Strategy: Awareness of visibility. 0.6 means safe from non-detectors.
    ALLIED_UNIT_CLOAK   = "allied_unit_cloak"


    # ==============================================================================
    # GROUP 2: ENEMY UNIT STATE (Layers 6-11)
    # Describes the targets. Mirror of Group 1, but for the opponent.
    # ==============================================================================

    # Layer 6: Enemy Locations
    # Logic: 1.0=Unit Present
    # Strategy: "Where do I click to attack?"
    ENEMY_UNIT_DENSITY  = "enemy_unit_density"

    # Layers 7-9: Enemy Tags
    # Logic: Same as Allied (Armor/Type/Special)
    # Strategy: Target Selection. "I am a Baneling, I should roll into Light units (0.5 on Layer 7)."
    ENEMY_UNIT_ARMOR    = "enemy_unit_armor"
    ENEMY_UNIT_TYPE     = "enemy_unit_type"
    ENEMY_UNIT_SPECIAL  = "enemy_unit_special"

    # Layer 10: Enemy Movement
    # Logic: 0.0=Ground, 1.0=Air, 0.5=Colossus
    # Strategy: "I am a Zealot, I cannot attack pixels that are 1.0 here."
    ENEMY_UNIT_FLYING   = "enemy_unit_flying"

    # Layer 11: Enemy Cloak
    # Logic: 0.6=Active Cloak (Visible because detected), 0.3=Capable (Ghost)
    # Strategy: High priority targets. If 0.6, they require detection nearby.
    ENEMY_UNIT_CLOAK    = "enemy_unit_cloak"


    # ==============================================================================
    # GROUP 3: VITALS (Layers 12-15)
    # Sustainability metrics.
    # ==============================================================================

    # Layers 12-13: Health Percentage
    # Logic: (HP + Shields) / Max. Range 0.0 to 1.0.
    # Strategy: Allied: Retreat low HP. Enemy: Focus fire low HP.
    ALLIED_HEALTH = "allied_health"
    ENEMY_HEALTH  = "enemy_health"

    # Layers 14-15: Energy Percentage
    # Logic: Current Energy / Max. Range 0.0 to 1.0.
    # Strategy: "Enemy HT has 1.0 energy -> Danger." "My Medivac has 0.0 energy -> Retreat."
    ALLIED_ENERGY = "allied_energy"
    ENEMY_ENERGY  = "enemy_energy"


    # ==============================================================================
    # GROUP 4: ALLIED OFFENSE MAPS (Layers 16-23)
    # Heatmaps of damage WE are projecting.
    # Logic: tanh(DPS / 50.0). White (1.0) = ~150 DPS.
    # Strategy: Helps AI maximize damage output by positioning units to overlap these fields.
    # ==============================================================================

    # Sub-Group: Ground Targets (Where we hit ground units)
    ALLIED_G_DPS_BASE    = "allied_g_dps_base"    # Raw damage (hits all ground)
    ALLIED_G_DPS_LIGHT   = "allied_g_dps_light"   # Bonus damage vs Light Ground (e.g. Hellion)
    ALLIED_G_DPS_ARMOR   = "allied_g_dps_armor"   # Bonus damage vs Armored Ground (e.g. Tank)
    ALLIED_G_DPS_MASSIVE = "allied_g_dps_massive" # Bonus damage vs Massive Ground

    # Sub-Group: Air Targets (Where we hit air units)
    ALLIED_A_DPS_BASE    = "allied_a_dps_base"    # Raw damage (hits all air)
    ALLIED_A_DPS_LIGHT   = "allied_a_dps_light"   # Bonus vs Light Air (e.g. Phoenix, Liberator)
    ALLIED_A_DPS_ARMOR   = "allied_a_dps_armor"   # Bonus vs Armored Air (e.g. Viking, Corruptor)
    ALLIED_A_DPS_MASSIVE = "allied_a_dps_massive" # Bonus vs Massive Air (e.g. Tempest)


    # ==============================================================================
    # GROUP 5: ENEMY THREAT MAPS (Layers 24-31)
    # Heatmaps of damage projected BY THE ENEMY.
    # Strategy: Evasion. "If I am Light, avoid Base+Light DPS. If Armored, avoid Base+Armor DPS."
    # ==============================================================================

    # Sub-Group: Ground Targets (Danger to our Ground Units)
    ENEMY_G_DPS_BASE     = "enemy_g_dps_base"
    ENEMY_G_DPS_LIGHT    = "enemy_g_dps_light"    # Danger zone for our Marines/Lings
    ENEMY_G_DPS_ARMOR    = "enemy_g_dps_armor"    # Danger zone for our Stalkers/Roaches
    ENEMY_G_DPS_MASSIVE  = "enemy_g_dps_massive"  # Danger zone for our Thors/Colossi

    # Sub-Group: Air Targets (Danger to our Air Units)
    ENEMY_A_DPS_BASE     = "enemy_a_dps_base"
    ENEMY_A_DPS_LIGHT    = "enemy_a_dps_light"    # Danger zone for Mutas/Banshees
    ENEMY_A_DPS_ARMOR    = "enemy_a_dps_armor"    # Danger zone for BCs/Corruptors
    ENEMY_A_DPS_MASSIVE  = "enemy_a_dps_massive"  # Danger zone for Capital Ships


    # ==============================================================================
    # GROUP 6: UTILITY & SUSTAIN (Layers 32-33)
    # ==============================================================================

    # Logic: tanh(HealingPerSecond / 50.0).
    # Strategy: Fight inside Allied fields. Kite enemies out of Enemy fields.
    ALLIED_HPS_FIELD     = "allied_hps_field"
    ENEMY_HPS_FIELD      = "enemy_hps_field"


    # ==============================================================================
    # GROUP 7: INTEL & ENVIRONMENT (Layers 34-37)
    # ==============================================================================

    # Layers 34-35: Detection
    # Logic: 1.0 inside detection radius.
    # Strategy: "Don't cloak here" (Enemy Detect) or "Cloaked enemies revealed here" (Allied Detect).
    ALLIED_DETECTION = "allied_detection"
    ENEMY_DETECTION  = "enemy_detection"

    # Layer 36: Terrain
    # Logic: 0.0=Wall/Blocked. 0.2 (Low Ground) to 1.0 (High Ground).
    # Strategy: Navigation. Ramps appear as gradients.
    TERRAIN_PATHABLE = "terrain_pathable"

    # Layer 37: Creep
    # Logic: 1.0=Creep, 0.0=No Creep.
    # Strategy: Zerg speed bonuses, Queen Transfuse eligibility.
    TERRAIN_CREEP    = "terrain_creep"


    # ==============================================================================
    # GROUP 8: PROTOCOL (Layers 38-39)
    # Instructions from the High-Level Commander Agent.
    # ==============================================================================

    # Layer 38: Assignment Mask
    # Logic: 1.0 on units this specific agent instance controls.
    # Strategy: "I am the Marine Agent, I ignore the Tanks marked 0.0".
    COMMAND_ASSIGNMENT = "command_assignment"

    # Layer 39: Objective Mask
    # Logic: 1.0 on the target area.
    # Strategy: "Go here / Defend here".
    COMMAND_OBJECTIVE  = "command_objective"

    # --- INDEX MAPPING ---
    LAYERS = {
        ALLIED_UNIT_DENSITY: 0, ALLIED_UNIT_ARMOR: 1, ALLIED_UNIT_TYPE: 2,
        ALLIED_UNIT_SPECIAL: 3, ALLIED_UNIT_FLYING: 4, ALLIED_UNIT_CLOAK: 5,

        ENEMY_UNIT_DENSITY: 6, ENEMY_UNIT_ARMOR: 7, ENEMY_UNIT_TYPE: 8,
        ENEMY_UNIT_SPECIAL: 9, ENEMY_UNIT_FLYING: 10, ENEMY_UNIT_CLOAK: 11,

        ALLIED_HEALTH: 12, ENEMY_HEALTH: 13, ALLIED_ENERGY: 14, ENEMY_ENERGY: 15,

        ALLIED_G_DPS_BASE: 16, ALLIED_G_DPS_LIGHT: 17, ALLIED_G_DPS_ARMOR: 18, ALLIED_G_DPS_MASSIVE: 19,
        ALLIED_A_DPS_BASE: 20, ALLIED_A_DPS_LIGHT: 21, ALLIED_A_DPS_ARMOR: 22, ALLIED_A_DPS_MASSIVE: 23,

        ENEMY_G_DPS_BASE: 24, ENEMY_G_DPS_LIGHT: 25, ENEMY_G_DPS_ARMOR: 26, ENEMY_G_DPS_MASSIVE: 27,
        ENEMY_A_DPS_BASE: 28, ENEMY_A_DPS_LIGHT: 29, ENEMY_A_DPS_ARMOR: 30, ENEMY_A_DPS_MASSIVE: 31,

        ALLIED_HPS_FIELD: 32, ENEMY_HPS_FIELD: 33,

        ALLIED_DETECTION: 34, ENEMY_DETECTION: 35, TERRAIN_PATHABLE: 36, TERRAIN_CREEP: 37,

        COMMAND_ASSIGNMENT: 38, COMMAND_OBJECTIVE: 39
    }

    DEFAULT_UNIT_CONTROL_LAYERS = list(LAYERS.keys())

    @classmethod
    def get_num_layers(cls) -> int:
        return 40
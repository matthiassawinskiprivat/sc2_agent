import sys
import os
import json
from sc2 import maps
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

from src.tools.unit_data_inclusion import (
    WEAPON_UPGRADE_BONUS_DB,
    WHITELIST,
    apply_unit_field_overrides,
    should_include_unit,
)

# --- 2. ATTRIBUTE BONUSES & TYPES (Manual Overrides) ---
# Type: 1=Ground, 2=Air, 3=Any
UNIT_WEAPON_OVERRIDES = {
    # PROTOSS
    "COLOSSUS": {"damage": 10.0, "type": 1, "bonuses": [{"type": 1, "bonus": 5.0}]},
    "IMMORTAL": {"damage": 20.0, "type": 1, "bonuses": [{"type": 2, "bonus": 30.0}]},
    "STALKER": {"damage": 13.0, "type": 3, "bonuses": [{"type": 2, "bonus": 5.0}]},
    "ADEPT": {"damage": 10.0, "type": 1, "bonuses": [{"type": 1, "bonus": 12.0}]},
    "VOIDRAY": {"damage": 6.0, "type": 3, "bonuses": [{"type": 2, "bonus": 4.0}]},
    "PHOENIX": {"damage": 5.0, "type": 2, "bonuses": [{"type": 1, "bonus": 5.0}]},  # Type 2 = Air Only
    "ORACLE": {"damage": 15.0, "type": 1, "bonuses": [{"type": 1, "bonus": 7.0}]},
    "HIGHTEMPLAR": {"damage": 4.0, "type": 1, "period": 1.25, "bonuses": []},

    # TERRAN
    "MARAUDER": {"damage": 10.0, "type": 1, "bonuses": [{"type": 2, "bonus": 10.0}]},
    "SIEGETANK": {"damage": 15.0, "type": 1, "bonuses": [{"type": 2, "bonus": 10.0}]},
    "SIEGETANKSIEGED": {"damage": 40.0, "type": 1, "bonuses": [{"type": 2, "bonus": 30.0}]},
    "HELLION": {"damage": 8.0, "type": 1, "bonuses": [{"type": 1, "bonus": 6.0}]},
    "HELLIONTANK": {"damage": 18.0, "type": 1, "bonuses": [{"type": 1, "bonus": 12.0}]},
    "VIKINGFIGHTER": {"damage": 10.0, "type": 2, "bonuses": [{"type": 2, "bonus": 4.0}]},
    "GHOST": {"damage": 10.0, "type": 3, "bonuses": [{"type": 1, "bonus": 10.0}]},
    "LIBERATOR": {"damage": 5.0, "type": 2, "bonuses": [{"type": 1, "bonus": 2.0}]},

    # ZERG
    "BANELING": {"damage": 20.0, "type": 1, "range": 0.25, "bonuses": [{"type": 1, "bonus": 15.0}]},
    "LURKERMP": {"damage": 20.0, "type": 1, "bonuses": [{"type": 2, "bonus": 10.0}]},
    "SPINECRAWLER": {"damage": 25.0, "type": 1, "bonuses": [{"type": 2, "bonus": 5.0}]},
    "SPORECRAWLER": {"damage": 15.0, "type": 2, "bonuses": []},
    "ULTRALISK": {"damage": 35.0, "type": 1, "bonuses": []},
    "ROACH": {"damage": 16.0, "type": 1, "bonuses": []},
}

# --- 3. DETECTION RANGES ---
DETECTOR_RANGES = {
    "MISSILETURRET": 11.0, "SPORECRAWLER": 11.0, "PHOTONCANNON": 11.0,
    "OBSERVER": 11.0, "OBSERVERSIEGEMODE": 11.0, "OVERSEER": 11.0,
    "OVERSEERSIEGEMODE": 11.0, "RAVEN": 11.0, "MOTHERSHIP": 11.0
}


class DataExtractorBot(BotAI):
    async def on_step(self, iteration):
        client = self.client
        if not client: sys.exit(1)

        if iteration == 0: return
        if iteration == 1:
            game_data = await client.get_game_data()
            for unit_id, unit_data in game_data.units.items():
                name = unit_data.name.upper()
                # Spawn combat + whitelist units so radius_map is populated for
                # weaponless casters/spawners (e.g. SWARMHOSTMP / INFESTOR).
                if name in WEAPON_UPGRADE_BONUS_DB or name in WHITELIST:
                    try:
                        await client.debug_create_unit([[UnitTypeId(unit_id), 1, self.game_info.map_center, 1]])
                    except:
                        pass
            return

        if iteration == 10:
            print("Extracting COMPLETE Game Data...")
            game_data = await client.get_game_data()
            radius_map = {u.type_id.value: u.radius for u in self.all_units}
            unit_db = {}

            for unit_id, unit_data in game_data.units.items():
                if not unit_data.name: continue
                name = unit_data.name.upper()

                radius = radius_map.get(unit_id, 0.0)
                if radius == 0.0:
                    proto = getattr(unit_data, "_proto", None)
                    if proto: radius = getattr(proto, "unit_radius", 0.0)

                footprint_radius = unit_data.footprint_radius
                if footprint_radius is not None and footprint_radius <= 0:
                    footprint_radius = None

                supply = 0.0
                proto = getattr(unit_data, "_proto", None)
                if proto: supply = getattr(proto, "food_required", 0.0)

                entry = {
                    "id": unit_id,
                    "name": name,
                    "race": str(unit_data.race),
                    "cost": {"minerals": unit_data.cost.minerals, "vespene": unit_data.cost.vespene, "supply": supply},
                    "radius": radius,
                    "sight": getattr(unit_data, "sight_range", 9.0),
                    "detect": DETECTOR_RANGES.get(name, 0.0),
                    "weapons": []
                }
                if footprint_radius is not None:
                    entry["footprint_radius"] = float(footprint_radius)

                raw_weapons = getattr(unit_data, "weapons", [])
                if not raw_weapons and proto and hasattr(proto, "weapons"):
                    raw_weapons = proto.weapons

                override = UNIT_WEAPON_OVERRIDES.get(name)

                # A. Standard Extraction (with Override Patching)
                if raw_weapons:
                    for idx, w in enumerate(raw_weapons):
                        w_type = getattr(w, "type", 3)
                        if hasattr(w_type, "value"): w_type = w_type.value

                        period = getattr(w, "speed", 0.0)
                        if period == 0 and hasattr(w, "period"): period = getattr(w, "period", 0.0)

                        w_range = getattr(w, "range", 0.0)
                        if override and idx < 1:  # Assuming primary weapon override
                            damage = override["damage"]
                            bonuses = override["bonuses"]
                            # Important: If override specifies type, use it. Otherwise use API type.
                            if "type" in override:
                                w_type = override["type"]
                            if "range" in override:
                                w_range = override["range"]
                        else:
                            damage = getattr(w, "damage", 0.0)
                            bonuses = []
                            if hasattr(w, "damage_bonus"):
                                for b in w.damage_bonus:
                                    attr = getattr(b, "attribute", 0)
                                    val = getattr(b, "bonus", 0.0)
                                    bonuses.append({"type": attr, "bonus": val})

                        entry["weapons"].append({
                            "type": w_type,
                            "damage": damage,
                            "attacks": getattr(w, "attacks", 1),
                            "period": period,
                            "range": w_range,
                            "upgrade_bonus": WEAPON_UPGRADE_BONUS_DB.get(name, 0.0),
                            "bonuses": bonuses
                        })

                # B. Virtual Weapon Injection (Fallback)
                elif override:
                    entry["weapons"].append({
                        "type": override.get("type", 3),  # Default to Any if not specified, but now all are specified
                        "damage": override["damage"],
                        "attacks": 1,
                        "period": override.get("period", 1.0),
                        "range": override.get("range", 6.0),
                        "upgrade_bonus": WEAPON_UPGRADE_BONUS_DB.get(name, 0.0),
                        "bonuses": override["bonuses"]
                    })

                if should_include_unit(name, bool(entry["weapons"])):
                    unit_db[name] = entry

            apply_unit_field_overrides(unit_db)

            output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data",
                                       "unit_data.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(unit_db, f, indent=2)
            print(f"Successfully saved {len(unit_db)} entities to {output_path}")
            try:
                await client.leave()
            except:
                pass
            sys.exit(0)


def main():
    try:
        run_game(maps.get("PlateauMicro_2"),
                 [Bot(Race.Terran, DataExtractorBot()), Computer(Race.Zerg, Difficulty.VeryEasy)], realtime=False)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

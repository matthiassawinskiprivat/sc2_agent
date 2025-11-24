"""
StarCraft 2 Unit Data Extractor - python-sc2 compatible
Extracts unit data without pysc2 dependency
"""

import json
import os
from typing import Dict, List, Any
from sc2.ids.unit_typeid import UnitTypeId

class UnitDataExtractor:
    """Extracts unit data from python-sc2 and static sources"""

    def __init__(self):
        self.unit_data = {}

    def extract_unit_data(self) -> Dict[str, Any]:
        """Extract unit data from python-sc2 UnitTypeId enum"""
        print("Extracting unit data from python-sc2...")

        # Extract from python-sc2
        sc2_data = self._extract_from_python_sc2()

        print(f"Final dataset contains {len(sc2_data)} units")
        return sc2_data

    def _extract_from_python_sc2(self) -> Dict[str, Any]:
        """Extract unit information from python-sc2"""
        unit_dict = {}

        for unit_type in UnitTypeId:
            unit_id = unit_type.value
            unit_name = unit_type.name

            if unit_id == 0 or unit_name == 'INVALID':
                continue

            unit_info = {
                'unit_id': unit_id,
                'name': unit_name,
                'race': self._get_unit_race(unit_name),
                'attributes': self._get_unit_attributes(unit_name),
                'weapons': self._get_unit_weapons(unit_name),
                'abilities': self._get_unit_abilities(unit_name),
                'health': self._get_unit_health(unit_name),
                'shield': self._get_unit_shield(unit_name),
                'energy': self._get_unit_energy(unit_name),
                'is_spell_caster': self._is_spell_caster(unit_name),
                'can_transform': self._can_transform(unit_name),
                'transform_abilities': self._get_transform_abilities(unit_name),
                'source': 'python-sc2'
            }

            unit_dict[unit_name] = unit_info

        return unit_dict

    def _get_unit_race(self, name: str) -> str:
        """Determine unit race"""
        if any(x in name for x in ['ZEALOT', 'STALKER', 'PROBE', 'VOIDRAY', 'CARRIER']):
            return "Protoss"
        elif any(x in name for x in ['MARINE', 'MARAUDER', 'SCV', 'SIEGETANK', 'BATTLECRUISER']):
            return "Terran"
        elif any(x in name for x in ['ZERGLING', 'DRONE', 'HYDRALISK', 'ULTRALISK']):
            return "Zerg"
        return "Unknown"

    def _get_unit_attributes(self, name: str) -> List[str]:
        """Get unit attributes"""
        attrs = []

        # Flying vs Ground
        if any(x in name for x in ['PHOENIX', 'MEDIVAC', 'MUTALISK', 'OVERLORD']):
            attrs.append("Flying")
        else:
            attrs.append("Ground")

        # Mechanical vs Biological
        if any(x in name for x in ['SCV', 'SIEGETANK', 'THOR', 'BATTLECRUISER']):
            attrs.append("Mechanical")
        elif any(x in name for x in ['ZERGLING', 'HYDRALISK', 'ULTRALISK']):
            attrs.append("Biological")

        return attrs

    def _get_unit_weapons(self, unit_name: str) -> List[Dict[str, Any]]:
        """Get weapon data for unit"""
        return self._get_weapon_data().get(unit_name, [])

    def _get_weapon_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Static weapon data - same as before"""
        return {
            "Marine": [{"type": "Ground", "damage": 6, "range": 5, "cooldown": 0.61, "aoe_radius": 0}],
            "Marauder": [{"type": "Ground", "damage": 20, "range": 6, "cooldown": 1.07, "aoe_radius": 0}],
            "SiegeTank": [{"type": "Ground", "damage": 40, "range": 7, "cooldown": 2.14, "aoe_radius": 0}],
            "SiegeTankSieged": [{"type": "Ground", "damage": 40, "range": 13, "cooldown": 2.14, "aoe_radius": 1.5}],
            "Thor": [{"type": "Ground", "damage": 30, "range": 7, "cooldown": 1.28}, {"type": "Air", "damage": 6, "range": 10, "cooldown": 2.14}],
            "VikingFighter": [{"type": "Air", "damage": 10, "range": 9, "cooldown": 1.28}],
            "Battlecruiser": [{"type": "Ground", "damage": 8, "range": 6, "cooldown": 0.45}],
            "Zealot": [{"type": "Ground", "damage": 8, "range": 0.1, "cooldown": 1.34}],
            "Stalker": [{"type": "Ground", "damage": 13, "range": 6, "cooldown": 1.34}, {"type": "Air", "damage": 13, "range": 6, "cooldown": 1.34}],
            "Immortal": [{"type": "Ground", "damage": 20, "range": 6, "cooldown": 1.45}],
            "Zergling": [{"type": "Ground", "damage": 5, "range": 0.1, "cooldown": 0.67}],
            "Hydralisk": [{"type": "Ground", "damage": 12, "range": 5, "cooldown": 0.59}, {"type": "Air", "damage": 12, "range": 5, "cooldown": 0.59}],
        }

    def _get_unit_abilities(self, name: str) -> List[str]:
        """Get abilities for unit"""
        if any(x in name for x in ['VIKING']):
            return ["AssaultMode", "FighterMode"]
        elif any(x in name for x in ['SIEGETANK']):
            return ["SiegeMode", "Unsiege"]
        elif any(x in name for x in ['THOR']):
            return ["ExplosiveMode", "HighImpactMode"]
        elif any(x in name for x in ['LURKER']):
            return ["Burrow", "Unburrow"]
        return []

    def _get_unit_health(self, name: str) -> int:
        """Get base health for unit"""
        health_data = {
            "Marine": 45, "Marauder": 125, "SiegeTank": 160, "Thor": 400,
            "VikingFighter": 125, "Battlecruiser": 550, "Zealot": 100,
            "Stalker": 80, "Immortal": 200, "Zergling": 35, "Hydralisk": 80,
        }
        return health_data.get(name, 50)

    def _get_unit_shield(self, name: str) -> int:
        """Get shield for Protoss units"""
        if any(x in name for x in ['ZEALOT', 'STALKER', 'IMMORTAL']):
            return 50
        return 0

    def _get_unit_energy(self, name: str) -> int:
        """Get energy for spell casters"""
        if any(x in name for x in ['GHOST', 'SENTRY', 'HIGHTEMPLAR', 'INFESTOR', 'VIPER']):
            return 200
        return 0

    def _is_spell_caster(self, name: str) -> bool:
        """Check if unit is a spell caster"""
        return any(x in name for x in ['GHOST', 'SENTRY', 'HIGHTEMPLAR', 'INFESTOR', 'VIPER'])

    def _can_transform(self, name: str) -> bool:
        """Check if unit can transform"""
        return any(x in name for x in ['VIKING', 'SIEGETANK', 'THOR', 'LURKER'])

    def _get_transform_abilities(self, name: str) -> List[str]:
        """Get transform abilities for unit"""
        if 'VIKING' in name:
            return ["AssaultMode", "FighterMode"]
        elif 'SIEGETANK' in name:
            return ["SiegeMode", "Unsiege"]
        elif 'THOR' in name:
            return ["ExplosiveMode", "HighImpactMode"]
        elif 'LURKER' in name:
            return ["Burrow", "Unburrow"]
        return []

    def save_unit_data(self, filename: str = None):
        """Save unit data to JSON file"""
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "../../data/unit_data.json")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        unit_data = self.extract_unit_data()

        with open(filename, 'w') as f:
            json.dump(unit_data, f, indent=2)

        print(f"Unit data saved to {filename}")
        return filename


# Keep compatibility stubs if other code references them
class ExternalDataFetcher:
    def fetch_from_spawning_tool(self): return {}
    def fetch_from_liquipedia(self): return {}
    def fetch_from_sc2replaystats(self): return {}
    def combine_unit_data(self, *args): return args[0] if args else {}

class StaticUnitDataProvider:
    @staticmethod
    def get_fallback_data(): return {}
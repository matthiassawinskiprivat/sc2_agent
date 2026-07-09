from typing import Dict, List, Optional, Set, Tuple

from sc2.position import Point2


class SquadRegistry:
    """
    Generalized squad membership and per-unit combat directives.
    Prototype extracted from CTF unit_command_map / unit_target_map patterns.
    """

    def __init__(self) -> None:
        self.squad_unit_tags: Dict[str, List[int]] = {}
        self.unit_command_map: Dict[int, str] = {}
        self.unit_target_map: Dict[int, Point2] = {}

    def clear(self) -> None:
        self.squad_unit_tags.clear()
        self.unit_command_map.clear()
        self.unit_target_map.clear()

    def register_squad(self, squad_id: str, unit_tags: List[int]) -> None:
        self.squad_unit_tags[squad_id] = list(unit_tags)

    def set_unit_directive(self, unit_tag: int, command: str, target: Optional[Point2] = None) -> None:
        self.unit_command_map[int(unit_tag)] = command
        if target is not None:
            self.unit_target_map[int(unit_tag)] = target

    def set_squad_directive(
        self, squad_id: str, command: str, target: Optional[Point2] = None
    ) -> None:
        for tag in self.squad_unit_tags.get(squad_id, []):
            self.set_unit_directive(tag, command, target)

    def get_directive(self, unit_tag: int) -> Tuple[Optional[str], Optional[Point2]]:
        tag = int(unit_tag)
        return self.unit_command_map.get(tag), self.unit_target_map.get(tag)

    def tags_for_squad(self, squad_id: str) -> List[int]:
        return list(self.squad_unit_tags.get(squad_id, []))

    def all_combat_tags(self) -> Set[int]:
        return set(self.unit_command_map.keys())

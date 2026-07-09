from enum import Enum
from typing import Dict, Iterable, List, Set


class WorkerRole(str, Enum):
    """Manager-assigned role for economy workers."""

    MINING = "mining"
    BUILD = "build"
    MICRO = "micro"


class WorkerRegistry:
    """
    Maps worker unit tags to roles. Manager is the sole writer; sub-agents read.
    """

    def __init__(self) -> None:
        self._roles: Dict[int, WorkerRole] = {}

    def clear(self) -> None:
        self._roles.clear()

    def assign(self, unit_tag: int, role: WorkerRole) -> None:
        self._roles[int(unit_tag)] = role

    def assign_many(self, unit_tags: Iterable[int], role: WorkerRole) -> None:
        for tag in unit_tags:
            self.assign(tag, role)

    def get_role(self, unit_tag: int) -> WorkerRole | None:
        return self._roles.get(int(unit_tag))

    def set_role(self, unit_tag: int, role: WorkerRole) -> None:
        self.assign(unit_tag, role)

    def remove(self, unit_tag: int) -> None:
        self._roles.pop(int(unit_tag), None)

    def tags_with_role(self, role: WorkerRole) -> List[int]:
        return sorted(tag for tag, r in self._roles.items() if r == role)

    def build_pool_tags(self) -> List[int]:
        return self.tags_with_role(WorkerRole.BUILD)

    def mining_tags(self) -> Set[int]:
        return set(self.tags_with_role(WorkerRole.MINING))

    def micro_tags(self) -> Set[int]:
        return set(self.tags_with_role(WorkerRole.MICRO))

    def reassign_to_micro(self, unit_tags: Iterable[int]) -> None:
        """Pull workers off economy tasks for combat support (e.g. all-in defense)."""
        self.assign_many(unit_tags, WorkerRole.MICRO)

    def all_tags(self) -> List[int]:
        return sorted(self._roles.keys())

    def snapshot(self) -> Dict[int, WorkerRole]:
        return dict(self._roles)

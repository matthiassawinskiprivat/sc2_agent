from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Set
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

class CommandType(Enum):
    """
    High-level instruction types from Manager to Sub-Agents.
    """
    NO_OP = auto()
    ATTACK = auto()    # Combat: Move/Attack a location
    DEFEND = auto()    # Combat: Hold ground at location
    BUILD = auto()     # Economy: Construct a building
    TRAIN = auto()     # Economy: Train a unit from a structure
    SCOUT = auto()     # Info: Move to location (ignoring enemies if possible)
    EXPAND = auto()    # Economy: Specialized Build for Bases (Nexus/CC/Hatchery)
    DROP = auto()      # Transport: Load and Drop units at location
    MINE = auto()      # Economy: Assign N workers near a region to mine minerals
    MINE_GAS = auto()  # Economy: Assign N workers near a region onto extractors
    INJECT = auto()    # Economy: Queen inject larva into a hatch/lair/hive


# Queen InjectLarva energy cost (SC2).
INJECT_ENERGY_COST = 25.0


@dataclass
class Command:
    """
    The Protocol Object passed from Manager Agent to Sub-Agents.

    Worker role assignment (mining / build / micro) is owned by WorkerRegistry,
    not embedded in Command. Manager updates the registry when emitting commands.

    ``MINE`` means \"assign up to ``worker_count`` drones near ``target_pos``\"
    (closest hatch / mineral line). ``worker_count is None`` means all workers.

    ``MINE_GAS`` means \"put up to ``worker_count`` mining-role drones on ready
    extractors near ``target_pos``\". ``worker_count`` 0 / None clears gas
    (pull off). Re-emitting replaces the standing gas directive (same pattern
    as ``MINE``). The manager decides when to saturate or pull off (e.g. bank
    100 gas for Metabolic Boost).

    ``TRAIN`` ``count`` is how many units of ``unit_type_id`` to produce before
    the command completes (``None`` / ``<=0`` => 1). One manager emission can
    request an army chunk; the train runtime keeps producing between emissions.

    ``INJECT`` is a macro command: the build sub-agent must select a queen and
    a townhall cell; it completes when InjectLarva is successfully issued.

    Commands are either executable when prerequisites hold (e.g. pool ready for
    TRAIN ZERGLING) or invalid — there is no urgency flag. Invalid manager
    emissions are a manager-level concern (negative reward), not a sub-agent flag.
    """
    command_type: CommandType

    # -- Payload --
    # For BUILD/TRAIN/EXPAND: structure or unit to produce (UnitTypeId).
    unit_type_id: Optional[UnitTypeId] = None

    # For ATTACK/DEFEND/BUILD/EXPAND/MINE/MINE_GAS/INJECT: Where?
    # Note: This might be a "Coarse" location (center of a 16x16 grid cell).
    # Sub-agents are responsible for finding precise local coordinates.
    target_pos: Optional[Point2] = None

    # For ATTACK: Focus fire specific enemy?
    target_unit_tag: Optional[int] = None

    # For MINE / MINE_GAS: how many drones near the region
    # (MINE: None / <=0 => all workers; MINE_GAS: None / <=0 => clear gas).
    worker_count: Optional[int] = None

    # For TRAIN: how many units to produce before the command completes
    # (None / <=0 => 1).
    count: Optional[int] = None

    # Assigned units for this command (set by CommandManager, not the Neural Net).
    # Optional override; MINE normally resolves WHO from worker_count + region.
    assigned_unit_tags: Set[int] = field(default_factory=set)

    # Optional squad id for ATTACK/DEFEND grouping.
    squad_id: Optional[str] = None

    @property
    def is_combat(self) -> bool:
        return self.command_type in {CommandType.ATTACK, CommandType.DEFEND, CommandType.SCOUT, CommandType.DROP}

    @property
    def is_macro(self) -> bool:
        return self.command_type in {
            CommandType.BUILD,
            CommandType.TRAIN,
            CommandType.EXPAND,
            CommandType.INJECT,
        }

    @property
    def is_economy(self) -> bool:
        return self.command_type in {CommandType.MINE, CommandType.MINE_GAS}

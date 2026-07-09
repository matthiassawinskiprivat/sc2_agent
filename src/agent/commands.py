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

@dataclass
class Command:
    """
    The Protocol Object passed from Manager Agent to Sub-Agents.

    Worker role assignment (mining / build / micro) is owned by WorkerRegistry,
    not embedded in Command. Manager updates the registry when emitting commands.
    """
    command_type: CommandType
    
    # -- Payload --
    # For BUILD/TRAIN/EXPAND: structure or unit to produce (UnitTypeId).
    unit_type_id: Optional[UnitTypeId] = None
    
    # For ATTACK/DEFEND/BUILD/EXPAND: Where?
    # Note: This might be a "Coarse" location (center of a 16x16 grid cell).
    # Sub-agents are responsible for finding precise local coordinates.
    target_pos: Optional[Point2] = None
    
    # For ATTACK: Focus fire specific enemy?
    target_unit_tag: Optional[int] = None
    
    # -- Execution Logic --
    # If True, Sub-Agent should interrupt tasks (e.g., pull mining SCVs).
    # If False, wait for idle resources.
    urgent: bool = True
    
    # Assigned units for this command (set by CommandManager, not the Neural Net)
    assigned_unit_tags: Set[int] = field(default_factory=set)

    @property
    def is_combat(self) -> bool:
        return self.command_type in {CommandType.ATTACK, CommandType.DEFEND, CommandType.SCOUT, CommandType.DROP}

    @property
    def is_macro(self) -> bool:
        return self.command_type in {CommandType.BUILD, CommandType.TRAIN, CommandType.EXPAND}

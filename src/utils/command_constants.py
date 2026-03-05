from typing import Dict

class CommandConstants:
    """
    Central mapping for High-Level Commands (Directives).
    Total slots: 10
    """
    # Indices
    SCOUT = 0
    ATTACK = 1
    DEFEND = 2
    DROP = 3
    
    # Mapping for YAML resolution
    DIRECTIVES = {
        "scout": SCOUT,
        "attack": ATTACK,
        "defend": DEFEND,
        "drop": DROP,
    }

    @classmethod
    def get_index(cls, name: str) -> int:
        """Returns the index for a given command name, default to 0 (Scout)."""
        if isinstance(name, int):
            return name
        return cls.DIRECTIVES.get(name.lower(), 0)

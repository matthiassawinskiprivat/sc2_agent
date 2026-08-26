"""Constants for the learned manager agent."""

from src.agent.commands import CommandType

# Manager action axes
MANAGER_COMMAND_TYPES = (
    CommandType.MINE,
    CommandType.BUILD,
    CommandType.TRAIN,
    CommandType.INJECT,
    CommandType.ATTACK,
    CommandType.DEFEND,
    CommandType.NO_OP,
)

MANAGER_MAP_RESOLUTION = 32
MANAGER_MACRO_VECTOR_DIM = 12
MANAGER_COUNT_BUCKETS = 8  # 0..7 → count = bucket (TRAIN/MINE worker_count)
MAX_MANAGER_UNIT_TYPES = 16

DEFAULT_MANAGER_UNIT_VOCAB = (
    "drone",
    "overlord",
    "queen",
    "zergling",
    "spawningpool",
    "extractor",
    "roachwarren",
    "roach",
    "noop",
)

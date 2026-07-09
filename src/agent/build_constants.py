"""Constants for the build-agent training stack."""

MAX_BUILD_WORKERS = 24
BUILD_MAP_RESOLUTION = 16
WORKER_FEAT_DIM = 8
MACRO_VECTOR_DIM = 8
COMMAND_VECTOR_DIM = 12
BUILD_PLACEMENT_CELLS = BUILD_MAP_RESOLUTION * BUILD_MAP_RESOLUTION

# UnitTypeId index space for command embedding (matches BuildNetwork cmd_embedding size).
MAX_UNIT_TYPE_INDEX = 200

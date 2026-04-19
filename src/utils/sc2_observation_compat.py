"""
python-sc2 vs SC2 client version skew.

Newer game builds (or ``debug_upgrade()``) can report ``upgrade_ids`` that are not yet
present in the installed ``UpgradeId`` enum; ``GameState`` would raise ``ValueError``.
Training does not depend on the exact upgrade set for observations, so unknown IDs
can be dropped before constructing ``GameState``.
"""


def strip_unknown_upgrade_ids_from_response_observation(response_observation) -> None:
    """Remove upgrade IDs from the live observation proto that ``UpgradeId`` cannot parse.

    Mutates ``response_observation`` in place (protobuf repeated field).
    """
    from sc2.ids.upgrade_id import UpgradeId

    try:
        upgrade_ids = response_observation.observation.raw_data.player.upgrade_ids
    except AttributeError:
        return

    kept = [uid for uid in upgrade_ids if _upgrade_id_is_known(uid, UpgradeId)]
    del upgrade_ids[:]
    upgrade_ids.extend(kept)


def _upgrade_id_is_known(uid: int, upgrade_id_enum) -> bool:
    try:
        upgrade_id_enum(uid)
        return True
    except ValueError:
        return False


def game_state_from_response_observation(response_observation, previous_observation=None):
    """Construct ``GameState`` after stripping unknown upgrade IDs."""
    strip_unknown_upgrade_ids_from_response_observation(response_observation)
    from sc2.game_state import GameState

    return GameState(response_observation, previous_observation=previous_observation)

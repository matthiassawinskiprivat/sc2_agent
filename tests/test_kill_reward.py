"""
Tests for per-kill bonus/penalty signals in CaptureTheFlagBot.calculate_squad_step_reward.

kill_reward (positive float) fires once per enemy mobile unit that dies.
kill_penalty (negative float in config) fires once per own mobile unit that dies.
Both stack additively on top of the continuous HP-delta reward.
"""
import pytest
from unittest.mock import MagicMock
from sc2.position import Point2

from src.agent.ctf_bot import CaptureTheFlagBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit(
    tag: int,
    hp: float = 100.0,
    shield: float = 0.0,
    is_structure: bool = False,
    type_id=None,
) -> MagicMock:
    u = MagicMock()
    u.tag = tag
    u.position = Point2((0.0, 0.0))
    u.owner_id = 1
    u.health = hp
    u.shield = shield
    u.is_structure = is_structure
    u.type_id = type_id
    return u


def _make_bot(kill_reward: float = 0.5, kill_penalty: float = -0.4) -> CaptureTheFlagBot:
    """Bot wired with only kill signals active (all other reward coefficients zero)."""
    bot = CaptureTheFlagBot(player_id=1, training_conf={
        "agent_architecture": "universal",
        "action_heads": ["move"],
        "categorical_heads": [],
        "rewards": {
            "kill_reward": kill_reward,
            "kill_penalty": kill_penalty,
            # zero-out all other signals so kill deltas are isolated
            "pylon_damage_reward": 0.0,
            "pylon_damage_penalty": 0.0,
            "damage_reward": 0.0,
            "damage_penalty": 0.0,
            "distance_multiplier": 0.0,
            "step_penalty": 0.0,
            "defender_step_reward": 0.0,
            "cooldown_retreat_scale": 0.0,
        },
    })
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = 1
    bot.prev_dist_per_squad = {}
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.prev_enemy_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    return bot


def _wire(bot, own_units, enemy_units):
    bot.player_id = 1
    bot.units = list(own_units)
    bot.enemy_units = list(enemy_units)

    def _struct_col():
        col = MagicMock()
        col.exists = False
        col.first = None
        return MagicMock(return_value=col)

    bot.structures = _struct_col()
    bot.enemy_structures = _struct_col()
    bot.is_visible = MagicMock(return_value=False)


# ---------------------------------------------------------------------------
# kill_reward — enemy dies
# ---------------------------------------------------------------------------

def test_kill_reward_fires_when_enemy_dies():
    bot = _make_bot(kill_reward=0.5, kill_penalty=0.0)
    enemy = _make_unit(tag=99)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[enemy])

    # Seed: 1 enemy alive last step
    bot.prev_enemy_unit_count[1] = 1

    # Now enemy is dead (empty list)
    _wire(bot, own_units=[own], enemy_units=[])
    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(0.5), f"Expected +0.5 kill_reward, got {reward}"


def test_kill_reward_scales_with_kills():
    """Two enemies die in one step → 2× kill_reward."""
    bot = _make_bot(kill_reward=0.5, kill_penalty=0.0)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[])

    bot.prev_enemy_unit_count[1] = 2  # two enemies were alive

    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(1.0), f"Expected +1.0 for two kills, got {reward}"


def test_kill_reward_not_fired_when_count_unchanged():
    bot = _make_bot(kill_reward=0.5, kill_penalty=0.0)
    enemy = _make_unit(tag=99)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[enemy])

    bot.prev_enemy_unit_count[1] = 1  # still 1 alive

    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(0.0), f"Expected 0.0 (no kill), got {reward}"


def test_kill_reward_not_fired_on_first_step():
    """On the very first step, prev count = 0 → no false kill even if enemies are present."""
    bot = _make_bot(kill_reward=0.5, kill_penalty=0.0)
    enemy = _make_unit(tag=99)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[enemy])

    # prev count stays at 0 (default)
    reward = bot.calculate_reward(perspective_player_id=1)

    # curr (1) > prev (0) → no kill signal
    assert reward == pytest.approx(0.0), f"Expected 0.0 on first step, got {reward}"


def test_kill_reward_zero_when_config_omitted():
    """kill_reward defaults to 0 when not in config → no reward on kill."""
    bot = CaptureTheFlagBot(player_id=1, training_conf={
        "agent_architecture": "universal",
        "action_heads": ["move"],
        "categorical_heads": [],
        "rewards": {
            "pylon_damage_reward": 0.0, "pylon_damage_penalty": 0.0,
            "damage_reward": 0.0, "damage_penalty": 0.0,
            "distance_multiplier": 0.0, "step_penalty": 0.0,
            "defender_step_reward": 0.0, "cooldown_retreat_scale": 0.0,
        },
    })
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = 1
    bot.prev_dist_per_squad = {}
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.prev_enemy_unit_count = {1: 2, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.enemy_pylon_pos = {1: None, 2: None}
    _wire(bot, own_units=[_make_unit(1)], enemy_units=[])

    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# kill_penalty — own unit dies
# ---------------------------------------------------------------------------

def test_kill_penalty_fires_when_own_unit_dies():
    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    own_survivor = _make_unit(tag=1)
    _wire(bot, own_units=[own_survivor], enemy_units=[])

    bot.prev_own_unit_count[1] = 2  # had 2 units

    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(-0.4), f"Expected -0.4 kill_penalty, got {reward}"


def test_kill_penalty_scales_with_losses():
    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    _wire(bot, own_units=[], enemy_units=[])

    bot.prev_own_unit_count[1] = 3  # lost all 3

    reward = bot.calculate_reward(perspective_player_id=1)

    assert reward == pytest.approx(-1.2), f"Expected -1.2 for three losses, got {reward}"


def test_kill_penalty_not_fired_when_count_unchanged():
    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[])

    bot.prev_own_unit_count[1] = 1

    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def test_counts_updated_after_step():
    """prev_enemy_unit_count and prev_own_unit_count are refreshed each step."""
    bot = _make_bot()
    own = [_make_unit(1), _make_unit(2)]
    enemies = [_make_unit(10), _make_unit(11), _make_unit(12)]
    _wire(bot, own_units=own, enemy_units=enemies)

    bot.calculate_reward(perspective_player_id=1)

    assert bot.prev_enemy_unit_count[1] == 3
    assert bot.prev_own_unit_count[1] == 2


# ---------------------------------------------------------------------------
# Structures excluded from kill counting
# ---------------------------------------------------------------------------

def test_structures_not_counted_as_kills():
    """Pylon deaths must not trigger kill_reward (they are structures)."""
    bot = _make_bot(kill_reward=0.5, kill_penalty=0.0)
    pylon = _make_unit(tag=200, is_structure=True)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[pylon])

    # Prev count is 0 (no mobile enemies), pylon is a structure → no kill
    bot.prev_enemy_unit_count[1] = 0

    reward = bot.calculate_reward(perspective_player_id=1)
    # curr mobile enemy count = 0 (pylon filtered), prev = 0 → no kill signal
    assert reward == pytest.approx(0.0)


def test_pylon_death_does_not_trigger_kill_reward():
    """When prev count had 1 pylon (filtered away), killing it gives 0 kill_reward."""
    bot = _make_bot(kill_reward=0.5, kill_penalty=0.0)
    own = _make_unit(tag=1)
    _wire(bot, own_units=[own], enemy_units=[])

    # Prev count was 0 (structures are always excluded)
    bot.prev_enemy_unit_count[1] = 0

    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Temporary summons excluded from own attrition penalties
# ---------------------------------------------------------------------------

def test_temporary_unit_not_counted_in_own_baseline():
    """Broodlings/locusts must not enter prev_own_unit_count / health baselines."""
    from sc2.ids.unit_typeid import UnitTypeId

    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    permanent = _make_unit(tag=1, hp=100.0, type_id=UnitTypeId.BROODLORD)
    broodling = _make_unit(tag=2, hp=30.0, type_id=UnitTypeId.BROODLING)
    locust = _make_unit(tag=3, hp=50.0, type_id=UnitTypeId.LOCUSTMP)
    _wire(bot, own_units=[permanent, broodling, locust], enemy_units=[])

    bot.calculate_reward(perspective_player_id=1)

    assert bot.prev_own_unit_count[1] == 1
    assert bot.prev_own_units_health[1] == pytest.approx(100.0)


def test_broodling_expiration_does_not_trigger_kill_penalty():
    """Timed broodlings expire on their own — must not fire kill_penalty."""
    from sc2.ids.unit_typeid import UnitTypeId

    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    permanent = _make_unit(tag=1, type_id=UnitTypeId.BROODLORD)
    broodling = _make_unit(tag=2, type_id=UnitTypeId.BROODLING)
    _wire(bot, own_units=[permanent, broodling], enemy_units=[])
    bot.calculate_reward(perspective_player_id=1)
    assert bot.prev_own_unit_count[1] == 1

    # Broodling expired / died; permanent army unchanged.
    _wire(bot, own_units=[permanent], enemy_units=[])
    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(0.0)
    assert bot.prev_own_unit_count[1] == 1


def test_locust_expiration_does_not_trigger_kill_penalty():
    from sc2.ids.unit_typeid import UnitTypeId

    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    permanent = _make_unit(tag=1, type_id=UnitTypeId.SWARMHOSTMP)
    locust = _make_unit(tag=2, type_id=UnitTypeId.LOCUSTMP)
    flying = _make_unit(tag=3, type_id=UnitTypeId.LOCUSTMPFLYING)
    _wire(bot, own_units=[permanent, locust, flying], enemy_units=[])
    bot.calculate_reward(perspective_player_id=1)
    assert bot.prev_own_unit_count[1] == 1

    _wire(bot, own_units=[permanent], enemy_units=[])
    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(0.0)


def test_broodling_damage_does_not_trigger_damage_penalty():
    """HP loss on a timed summon must not contribute to damage_penalty."""
    from sc2.ids.unit_typeid import UnitTypeId

    bot = CaptureTheFlagBot(player_id=1, training_conf={
        "agent_architecture": "universal",
        "action_heads": ["move"],
        "categorical_heads": [],
        "rewards": {
            "kill_reward": 0.0,
            "kill_penalty": 0.0,
            "pylon_damage_reward": 0.0,
            "pylon_damage_penalty": 0.0,
            "damage_reward": 0.0,
            "damage_penalty": 0.2,
            "distance_multiplier": 0.0,
            "step_penalty": 0.0,
            "defender_step_reward": 0.0,
            "cooldown_retreat_scale": 0.0,
        },
    })
    bot.unit_command_map = {}
    bot.unit_target_map = {}
    bot.active_unit_tag = 1
    bot.prev_dist_per_squad = {}
    bot.prev_pylon_health = {1: 20.0, 2: 20.0}
    bot.prev_enemy_units_health = {1: 0.0, 2: 0.0}
    bot.prev_own_units_health = {1: 0.0, 2: 0.0}
    bot.prev_enemy_unit_count = {1: 0, 2: 0}
    bot.prev_own_unit_count = {1: 0, 2: 0}
    bot.enemy_pylon_pos = {1: None, 2: None}

    permanent = _make_unit(tag=1, hp=100.0, type_id=UnitTypeId.BROODLORD)
    broodling = _make_unit(tag=2, hp=30.0, type_id=UnitTypeId.BROODLING)
    _wire(bot, own_units=[permanent, broodling], enemy_units=[])
    bot.calculate_reward(perspective_player_id=1)
    assert bot.prev_own_units_health[1] == pytest.approx(100.0)

    broodling.health = 5.0  # would be −25 own HP if wrongly included
    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(0.0)
    assert bot.prev_own_units_health[1] == pytest.approx(100.0)


def test_permanent_unit_death_still_triggers_kill_penalty_with_temps_present():
    """Losing a real army unit still penalizes even if temps are on the field."""
    from sc2.ids.unit_typeid import UnitTypeId

    bot = _make_bot(kill_reward=0.0, kill_penalty=-0.4)
    survivor = _make_unit(tag=1, type_id=UnitTypeId.HYDRALISK)
    doomed = _make_unit(tag=2, type_id=UnitTypeId.HYDRALISK)
    broodling = _make_unit(tag=99, type_id=UnitTypeId.BROODLING)
    _wire(bot, own_units=[survivor, doomed, broodling], enemy_units=[])
    bot.calculate_reward(perspective_player_id=1)
    assert bot.prev_own_unit_count[1] == 2

    _wire(bot, own_units=[survivor, broodling], enemy_units=[])
    reward = bot.calculate_reward(perspective_player_id=1)
    assert reward == pytest.approx(-0.4)
    assert bot.prev_own_unit_count[1] == 1

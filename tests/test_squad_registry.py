from sc2.position import Point2

from src.runtime.squad_registry import SquadRegistry


def test_squad_registry_directives():
    reg = SquadRegistry()
    reg.register_squad("defense", [10, 11])
    reg.set_squad_directive("defense", "attack", Point2((64.0, 64.0)))

    cmd, target = reg.get_directive(10)
    assert cmd == "attack"
    assert target == Point2((64.0, 64.0))
    assert reg.tags_for_squad("defense") == [10, 11]

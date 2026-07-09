from src.runtime.worker_registry import WorkerRegistry, WorkerRole


def test_worker_registry_roles_and_pools():
    reg = WorkerRegistry()
    reg.assign(10, WorkerRole.BUILD)
    reg.assign(20, WorkerRole.MINING)
    reg.assign(30, WorkerRole.MICRO)

    assert reg.get_role(10) == WorkerRole.BUILD
    assert reg.build_pool_tags() == [10]
    assert reg.mining_tags() == {20}
    assert reg.micro_tags() == {30}

    reg.reassign_to_micro([10, 20])
    assert reg.get_role(10) == WorkerRole.MICRO
    assert reg.get_role(20) == WorkerRole.MICRO
    assert reg.build_pool_tags() == []

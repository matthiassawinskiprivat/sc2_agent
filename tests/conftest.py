import pytest
import logging
from loguru import logger

@pytest.fixture
def caplog_loguru(caplog):
    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    yield caplog
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# CTF universal pipeline — SC2 smoke-test flag
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    try:
        parser.addoption(
            "--run-sc2",
            action="store_true",
            default=False,
            help="Run SC2 smoke tests (requires live SC2 installation)",
        )
    except ValueError:
        pass  # option already registered by another conftest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "sc2: mark test as requiring a live SC2 installation (--run-sc2)",
    )


def pytest_collection_modifyitems(config, items):
    try:
        run_sc2 = config.getoption("--run-sc2")
    except ValueError:
        run_sc2 = False
    if not run_sc2:
        skip_sc2 = pytest.mark.skip(
            reason="SC2 smoke tests skipped (pass --run-sc2 to enable)"
        )
        for item in items:
            if item.get_closest_marker("sc2"):
                item.add_marker(skip_sc2)

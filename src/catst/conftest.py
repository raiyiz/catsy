"""Shared pytest fixtures and test-run options."""

import pytest

from .states import GaussianOperations


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="run opt-in visual tests and enable plots in plot-aware tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "visual: opt-in tests that open or render plots")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        return
    deselected = [item for item in items if "visual" in item.keywords]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item not in deselected]


@pytest.fixture
def plot_enabled(request):
    """Whether the current test run explicitly opted into plotting."""
    return request.config.getoption("--plot")


@pytest.fixture
def single_mode_vacuum():
    return GaussianOperations.create_vacuum(("a",))


@pytest.fixture
def two_mode_vacuum():
    return GaussianOperations.create_vacuum(("a", "b"))

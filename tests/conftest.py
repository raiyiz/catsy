"""Shared pytest fixtures and options for the catsy test suite."""

import pytest

from catsy import GaussianState


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="show tests marked visual or optional diagnostic plots",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "visual: test intended for interactive plots")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        return
    skip_visual = pytest.mark.skip(reason="visual test; rerun with --plot")
    for item in items:
        if "visual" in item.keywords:
            item.add_marker(skip_visual)


@pytest.fixture
def single_mode_vacuum():
    return GaussianState.vacuum(("a",))


@pytest.fixture
def two_mode_vacuum():
    return GaussianState.vacuum(("a", "b"))


@pytest.fixture
def plot_enabled(request):
    return request.config.getoption("--plot")

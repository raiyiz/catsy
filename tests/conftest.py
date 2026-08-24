"""Shared pytest fixtures and options for the catsy test suite."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pytest

PLOT_PAUSE_SECONDS = 2.0

if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("GITLAB_CI") == "true":
    matplotlib.use("Agg")

from catsy.gaussian import GaussianState


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="run tests marked visualize and display their figures locally",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "visualize: test that exercises interactive visualizations"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        return

    skip_visualize = pytest.mark.skip(reason="visualization test; rerun with --plot")
    for item in items:
        if "visualize" in item.keywords:
            item.add_marker(skip_visualize)


@pytest.fixture
def single_mode_vacuum():
    return GaussianState.vacuum(("a",))


@pytest.fixture
def two_mode_vacuum():
    return GaussianState.vacuum(("a", "b"))


@pytest.fixture
def plot_enabled(request):
    """Whether visualization tests are enabled by ``--plot``."""
    return bool(request.config.getoption("--plot"))


@pytest.fixture
def show_plots(plot_enabled):
    """Whether visualization figures should be shown interactively."""
    in_ci = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("GITLAB_CI") == "true"
    return plot_enabled and not in_ci


@pytest.fixture(autouse=True)
def manage_visual_figures(request, show_plots):
    """Display visualization figures briefly, then close them."""
    if request.node.get_closest_marker("visualize") is None:
        yield
        return

    yield

    if show_plots:
        plt.show(block=False)
        plt.pause(PLOT_PAUSE_SECONDS)
    plt.close("all")

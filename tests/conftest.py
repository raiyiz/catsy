"""Shared pytest fixtures and options for the catsy test suite."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pytest

if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("GITLAB_CI") == "true":
    matplotlib.use("Agg")

from catsy.gaussian import GaussianState


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
    """Whether visual tests are enabled by the ``--plot`` option."""
    return bool(request.config.getoption("--plot"))


@pytest.fixture
def show_plots(plot_enabled):
    """Whether visual figures should be displayed interactively."""
    in_ci = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("GITLAB_CI") == "true"
    return plot_enabled and not in_ci


@pytest.fixture(autouse=True)
def manage_visual_figures(request, show_plots):
    """Display and close figures created by visual tests."""
    is_visual = request.node.get_closest_marker("visual") is not None
    if not is_visual:
        yield
        return

    yield

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    plt.close("all")

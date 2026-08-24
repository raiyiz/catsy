"""Shared pytest fixtures and options for the catsy test suite."""

import os

import matplotlib
import pytest

PLOT_PAUSE_SECONDS = 2.0
IN_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("GITLAB_CI") == "true"

if IN_CI:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

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


@pytest.fixture(autouse=True)
def manage_visual_figures(request, monkeypatch):
    """Own Matplotlib display and cleanup for every visualization test."""
    if request.node.get_closest_marker("visualize") is None:
        yield
        return

    def suppress_show(*args, **kwargs):
        """Prevent test code or library helpers from owning display policy."""
        return None

    monkeypatch.setattr(plt, "show", suppress_show)

    yield

    if request.config.getoption("--plot") and not IN_CI and plt.get_fignums():
        plt.show = suppress_show
        # Call the original backend-level show through Matplotlib's FigureManager
        # API so test/library calls to pyplot.show() remain centrally controlled.
        managers = [manager for manager in plt._pylab_helpers.Gcf.get_all_managers()]
        for manager in managers:
            manager.show()
        plt.pause(PLOT_PAUSE_SECONDS)

    plt.close("all")

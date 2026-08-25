"""Shared pytest fixtures and options for the catsy test suite."""

import os
import time
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from catsy.gaussian import GaussianState

IN_CI = os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("GITLAB_CI") == "true"

if IN_CI:
    matplotlib.use("Agg")

PLOT_PAUSE_SECONDS = 2.0


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        help="run tests marked visualize and display their figures locally",
    )
    parser.addoption(
        "--plot-pause",
        action="store",
        type=float,
        default=PLOT_PAUSE_SECONDS,
        help=f"seconds to pause between displayed visualization tests (default: {PLOT_PAUSE_SECONDS})",
    )
    parser.addoption(
        "--timings",
        action="store_true",
        help="report the slowest test durations at the end of the test session",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "visualize: test that exercises interactive visualizations"
    )
    config.addinivalue_line(
        "markers", "timing(max_seconds=...): enforce a wall-clock runtime budget"
    )
    if config.getoption("--plot-pause") < 0:
        raise pytest.UsageError("--plot-pause must be non-negative")
    config._catsy_timings = []


def pytest_collection_modifyitems(config, items):
    if config.getoption("--plot"):
        return

    skip_visualize = pytest.mark.skip(reason="visualization test; rerun with --plot")
    for item in items:
        if "visualize" in item.keywords:
            item.add_marker(skip_visualize)


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--timings"):
        return

    timings = sorted(
        session.config._catsy_timings,
        key=lambda entry: entry[1],
        reverse=True,
    )
    if not timings:
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is None:
        return

    terminal.write_sep("-", "catsy test timings")
    for nodeid, duration in timings[:20]:
        terminal.write_line(f"{duration:8.3f}s  {nodeid}")


def _assert_no_empty_axes(figure: Figure) -> None:
    """Every non-colorbar axis should contain at least one drawable artist."""
    for ax in figure.axes:
        if getattr(ax, "_colorbar", None) is not None:
            continue
        assert ax.lines or ax.patches or ax.collections or ax.images or ax.texts


def _assert_layout_can_render(figure: Figure) -> None:
    """Exercise the actual Matplotlib layout engine used by CI/savefig."""
    figure.canvas.draw()
    for ax in figure.axes:
        bbox = ax.get_window_extent()
        assert bbox.width > 0
        assert bbox.height > 0


@pytest.fixture
def assert_no_empty_axes() -> Callable[[Figure], None]:
    """Return the shared structural assertion for figure contents."""
    return _assert_no_empty_axes


@pytest.fixture
def assert_layout_can_render() -> Callable[[Figure], None]:
    """Return the shared structural assertion for rendered figure geometry."""
    return _assert_layout_can_render


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
def measure_test_timing(request):
    """Record wall-clock duration and enforce explicit timing budgets."""
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    request.config._catsy_timings.append((request.node.nodeid, duration))

    marker = request.node.get_closest_marker("timing")
    if marker is None:
        return

    max_seconds = marker.kwargs.get("max_seconds")
    if max_seconds is None:
        raise pytest.UsageError("timing marker requires max_seconds=...")
    if not isinstance(max_seconds, (int, float)) or max_seconds <= 0:
        raise pytest.UsageError("timing max_seconds must be a positive number")
    if duration > max_seconds:
        pytest.fail(
            f"test exceeded timing budget: {duration:.3f}s > {max_seconds:.3f}s",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def manage_visual_figures(request, monkeypatch):
    """Own Matplotlib display and cleanup for every visualization test."""
    if request.node.get_closest_marker("visualize") is None:
        yield
        return

    original_show = plt.show

    def suppress_show(*args, **kwargs):
        """Keep display policy in this fixture rather than individual tests."""
        return None

    monkeypatch.setattr(plt, "show", suppress_show)

    yield

    # --plot-pause is validated once in pytest_configure, so it's safe to
    # trust here; figure cleanup below always runs regardless of --plot.
    if request.config.getoption("--plot") and not IN_CI and plt.get_fignums():
        pause = request.config.getoption("--plot-pause")
        original_show(block=False)
        plt.pause(pause)

    plt.close("all")

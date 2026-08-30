import json

import numpy as np
import qutip as qt

from catsy.gaussian import GaussianState
from examples.complex_example import PLOT_STEMS, execution_stages
from scripts.sync_complex_execution_stages import (
    _load_stages,
    _render_pipeline,
    _render_stages,
)


def test_execution_stage_trace_preserves_fock_operations(tmp_path):
    journal = {
        "metadata": {
            "custom": {
                "execution_stages": [
                    {
                        "id": "gaussian_state_preparation",
                        "title": "Gaussian preparation",
                        "category": "gaussian",
                    },
                    {
                        "id": "photon_subtraction",
                        "title": "Photon subtraction",
                        "category": "fock",
                    },
                    {
                        "id": "photon_addition",
                        "title": "Photon addition",
                        "category": "fock",
                    },
                ]
            }
        }
    }
    path = tmp_path / "entry.json"
    path.write_text(json.dumps(journal), encoding="utf-8")

    stages = _load_stages(path)
    pipeline = _render_pipeline(stages)

    assert [stage["id"] for stage in stages] == [
        "gaussian_state_preparation",
        "photon_subtraction",
        "photon_addition",
    ]
    assert "Photon subtraction" in pipeline
    assert "Photon addition" in pipeline
    assert pipeline.count('class="pipeline-step"') == 3


def _cheap_execution_stages() -> list[dict[str, object]]:
    """Call execution_stages() with the cheapest inputs it actually reads.

    circuit/final_state/heterodyne_state are accepted but never read by the
    function body, so real objects would just be wasted setup here.
    """
    cat = subtracted = added = qt.fock_dm(2, 0)
    mzi_scan = {"theta": np.linspace(0.0, 1.0, 3)}
    homodyne_state = GaussianState.vacuum(("a",))
    return execution_stages(
        circuit=None,  # type: ignore[arg-type]
        final_state=None,  # type: ignore[arg-type]
        cat=cat,
        subtracted=subtracted,
        added=added,
        mzi_scan=mzi_scan,  # type: ignore[arg-type]
        homodyne_state=homodyne_state,
        heterodyne_state=None,  # type: ignore[arg-type]
    )


def test_execution_stages_plots_exactly_match_plot_experiment():
    """Regression guard: every plot plot_experiment() saves must be reachable
    from the explorer, and every "plot" a stage references must actually be
    generated. Catches a stage/plot list drifting apart -- e.g. a plot added
    to plot_experiment() but never wired into a stage, which
    sync_complex_execution_stages.py would then silently drop from the
    published page once it overwrites the built-in stage list with this
    (journal-sourced) one.
    """
    stages = _cheap_execution_stages()
    referenced_plots = {stage["plot"] for stage in stages if stage.get("plot")}
    assert referenced_plots == set(PLOT_STEMS)


def test_execution_stages_have_required_fields_and_unique_ids():
    stages = _cheap_execution_stages()
    ids = [stage["id"] for stage in stages]
    assert len(ids) == len(set(ids)), "execution_stages() has duplicate stage ids"
    for stage in stages:
        assert {"id", "title", "category", "insight"} <= stage.keys()
        assert stage["category"] in {"gaussian", "fock", "interferometer", "measurement"}


def test_sync_prefers_stage_specific_insight_over_generic_note(tmp_path):
    stages = [
        {
            "id": "gaussian_state_preparation",
            "title": "Gaussian preparation",
            "category": "gaussian",
            "insight": "A specific physical remark carried through from the example.",
        },
        {
            "id": "photon_subtraction",
            "title": "Photon subtraction",
            "category": "fock",
        },
    ]
    rendered = _render_stages(stages, tmp_path)

    assert "A specific physical remark carried through from the example." in rendered
    assert "Why it matters" in rendered
    # The stage without "insight" still gets a callout, just the generic one.
    assert "Recorded execution" in rendered

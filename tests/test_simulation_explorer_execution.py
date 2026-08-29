import json

from scripts.sync_complex_execution_stages import _load_stages, _render_pipeline


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
    assert pipeline.count("class=\"pipeline-step\"") == 3

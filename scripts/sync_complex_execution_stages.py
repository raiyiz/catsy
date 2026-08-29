"""Sync the complex Simulation Explorer with the execution journal.

The rich report has a presentation template, but the execution journal is the
source of truth for which stages actually ran and in what order. This script
replaces the generated pipeline and stage cards with the stage trace recorded
by ``examples/complex_example.py``.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

OUTPUT_ROOT = Path("_site")
RUN_ROOT = Path("runs/complex_circuit")
CATEGORY_LABELS = {
    "gaussian": "Gaussian",
    "fock": "Fock",
    "interferometer": "Interferometer",
    "measurement": "Measurement",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _find_journal(run_root: Path) -> Path:
    journals = sorted(run_root.glob("entry_*.json"))
    if not journals:
        raise SystemExit(f"No journal entry found below {run_root}")
    return journals[-1]


def _load_stages(journal_path: Path) -> list[dict[str, object]]:
    data = json.loads(journal_path.read_text(encoding="utf-8"))
    stages = data.get("metadata", {}).get("custom", {}).get("execution_stages")
    if not isinstance(stages, list) or not stages:
        raise SystemExit("Journal entry does not contain execution_stages metadata")
    normalized: list[dict[str, object]] = []
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise SystemExit(f"execution stage {index} is not an object")
        required = {"id", "title", "category"}
        missing = required - stage.keys()
        if missing:
            raise SystemExit(f"execution stage {index} is missing {sorted(missing)}")
        category = str(stage["category"])
        if category not in CATEGORY_LABELS:
            raise SystemExit(f"Unknown execution stage category: {category!r}")
        normalized.append(stage)
    return normalized


def _render_pipeline(stages: list[dict[str, object]]) -> str:
    cards = []
    for number, stage in enumerate(stages, start=1):
        category = str(stage["category"])
        cards.append(
            f'<a class="pipeline-step" href="#stage-{esc(number)}" '
            f'style="--step-color:var(--{esc(category)})">'
            f'<span class="n">{number:02d}</span>'
            f'<strong>{esc(stage["title"])}</strong>'
            f'<small>{esc(CATEGORY_LABELS[category])}</small></a>'
        )
    return "<div class=\"pipeline\">" + "".join(cards) + "</div>"


def _render_stages(stages: list[dict[str, object]], run_root: Path) -> str:
    cards = []
    for number, stage in enumerate(stages, start=1):
        category = str(stage["category"])
        plot_stem = stage.get("plot")
        plot = None
        if plot_stem:
            plot = next((p for p in (run_root / "plots").glob("*.png") if p.stem == str(plot_stem)), None)

        if plot:
            relative = esc(plot.relative_to(run_root).as_posix())
            visual = (
                f'<button class="stage-plot" data-src="{relative}" '
                f'data-title="{esc(stage["title"])}" type="button">'
                f'<img src="{relative}" alt="{esc(stage["title"])}" loading="lazy"></button>'
            )
        else:
            visual = '<div class="stage-plot empty"><span>—</span><small>response recorded in journal, not plotted</small></div>'

        description = str(stage.get("description", ""))
        inspect = str(stage.get("inspect", ""))
        result = str(stage.get("result", ""))
        cards.append(
            f'<article class="stage" data-category="{esc(category)}" id="stage-{number}">'
            f'<div class="stage-index" style="--stage-color:var(--{esc(category)})">'
            f'<span class="num">{number:02d}</span>'
            f'<span class="cat">{esc(CATEGORY_LABELS[category])}</span></div>'
            f'{visual}'
            f'<div class="stage-copy" style="--stage-color:var(--{esc(category)})">'
            f'<h3>{esc(stage["title"])}</h3><p>{esc(description)}</p>'
            f'<div class="callout"><span class="lbl">Look for</span><span>{esc(inspect)}</span></div>'
            f'<div class="callout"><span class="lbl">Diagnostic</span><span>{esc(result)}</span></div>'
            f'<div class="insight"><strong>Recorded execution</strong>'
            f'Stage {number:02d} is an execution record from the journal, not a manually added display-only step.</div>'
            f'</div></article>'
        )
    return '<div class="stages">' + "".join(cards) + '</div>'


def sync(report_path: Path, journal_path: Path, run_root: Path) -> None:
    text = report_path.read_text(encoding="utf-8")
    stages = _load_stages(journal_path)

    text, pipeline_count = re.subn(
        r'<div class="pipeline">.*?</div>',
        _render_pipeline(stages),
        text,
        count=1,
        flags=re.S,
    )
    if pipeline_count != 1:
        raise SystemExit("Could not locate generated pipeline in report")

    text, stage_count = re.subn(
        r'<div class="stages">.*?</div>\s*</div>\s*</section>',
        _render_stages(stages, run_root) + "\n</div></section>",
        text,
        count=1,
        flags=re.S,
    )
    if stage_count != 1:
        raise SystemExit("Could not locate generated stage list in report")

    text = text.replace(
        "<p>Each step links to the diagnostic and its physical interpretation below.</p>",
        "<p>Stages are read from the execution journal, so this map follows what the simulation actually ran.</p>",
        1,
    )
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    commit = os.environ.get("REPORT_COMMIT")
    if not commit:
        raise SystemExit("REPORT_COMMIT is required")
    site_run_root = OUTPUT_ROOT / "complex-example" / "runs" / commit
    report_path = site_run_root / "index.html"
    if not report_path.exists():
        raise SystemExit(f"Missing generated report: {report_path}")
    journal_path = _find_journal(site_run_root)
    sync(report_path, journal_path, site_run_root)


if __name__ == "__main__":
    main()

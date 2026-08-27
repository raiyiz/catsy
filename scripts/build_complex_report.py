"""Build a commit-addressed static HTML report for the complex example."""

from __future__ import annotations

import html
import os
from pathlib import Path


OUTPUT_ROOT = Path("_site")
RUN_ROOT = Path("runs/complex_circuit")


def build_run_page(site_run_root: Path, commit: str) -> None:
    short_commit = commit[:12]
    plots = sorted((site_run_root / "plots").glob("*.png"))
    journals = sorted(
        path for path in site_run_root.rglob("*") if path.suffix.lower() in {".json", ".jsonl"}
    )

    cards = []
    for plot in plots:
        relative = plot.relative_to(site_run_root).as_posix()
        title = plot.stem.replace("_", " ").title()
        cards.append(
            f'<figure><a href="{html.escape(relative)}">'
            f'<img src="{html.escape(relative)}" alt="{html.escape(title)}"></a>'
            f"<figcaption>{html.escape(title)}</figcaption></figure>"
        )

    journal_links = []
    for journal in journals:
        relative = journal.relative_to(site_run_root).as_posix()
        journal_links.append(
            f'<li><a href="{html.escape(relative)}">{html.escape(relative)}</a></li>'
        )

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catsy complex example — {short_commit}</title>
<style>
body {{ max-width: 1200px; margin: 0 auto; padding: 2rem; font-family: system-ui, sans-serif; line-height: 1.5; }}
h1, h2 {{ line-height: 1.2; }}
.meta {{ color: #555; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.25rem; }}
figure {{ margin: 0; }}
figure img {{ display: block; width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
figcaption {{ padding: .5rem 0; font-weight: 600; }}
a {{ color: inherit; }}
code {{ background: #f3f3f3; padding: .1em .3em; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Catsy complex example</h1>
<p class="meta">Commit <code>{html.escape(commit)}</code></p>
<p>Three-mode Gaussian preparation, heralded Fock processing, lossy Mach–Zehnder interferometry, and homodyne/heterodyne readout.</p>
<h2>Plots</h2>
<div class="grid">{''.join(cards) or '<p>No plots were generated.</p>'}</div>
<h2>Journal</h2>
<ul>{''.join(journal_links) or '<li>No journal files were generated.</li>'}</ul>
</body>
</html>
"""
    (site_run_root / "index.html").write_text(report, encoding="utf-8")


def main() -> None:
    if not RUN_ROOT.exists():
        raise SystemExit(f"Missing complex-example output: {RUN_ROOT}")

    commit = os.environ.get("GITHUB_SHA", "unknown")
    base_path = f"runs/{commit}"
    site_run_root = OUTPUT_ROOT / base_path
    site_run_root.parent.mkdir(parents=True, exist_ok=True)

    import shutil

    shutil.copytree(RUN_ROOT, site_run_root, dirs_exist_ok=True)
    build_run_page(site_run_root, commit)

    runs_root = OUTPUT_ROOT / "runs"
    run_dirs = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    ) if runs_root.exists() else []
    links = [
        f'<li><a href="runs/{html.escape(path.name)}/">{html.escape(path.name)}</a></li>'
        for path in run_dirs
    ]

    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Catsy complex-example reports</title>
<style>body {{ max-width: 900px; margin: 0 auto; padding: 2rem; font-family: system-ui, sans-serif; line-height: 1.5; }}</style>
</head><body>
<h1>Catsy complex-example reports</h1>
<p>Commit-addressed CI reports. Each run contains its plots and journal output.</p>
<ul>{''.join(links) or '<li>No reports yet.</li>'}</ul>
</body></html>
"""
    (OUTPUT_ROOT / "index.html").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()

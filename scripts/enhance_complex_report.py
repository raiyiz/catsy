"""Refine the generated complex-example Pages report for human exploration."""

from __future__ import annotations

import html
import os
import re
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_ROOT = Path("_site")
RUN_ROOT = Path("runs/complex_circuit")

INSIGHTS = {
    "01": (
        "State preparation",
        "The phase-space view shows the prepared signal mode before non-Gaussian processing.",
    ),
    "02": (
        "Covariance structure",
        "Read the diagonal as single-mode variances and the off-diagonal blocks as inter-mode correlations.",
    ),
    "03": (
        "Correlation structure",
        "This diagnostic makes the Gaussian coupling visible: correlated quadratures indicate information shared between modes.",
    ),
    "04": (
        "Non-Gaussian signature",
        "The Wigner function should reveal the two-lobed cat structure and its interference fringes.",
    ),
    "05": (
        "Fock-space evidence",
        "Use the photon-number structure to see how the even-parity cat is represented beyond a Gaussian description.",
    ),
    "06": (
        "Heralded subtraction",
        "Loss and detector efficiency make this a realistic conditional operation rather than an ideal annihilation.",
    ),
    "07": (
        "Heralded addition",
        "Compare this state with subtraction to see how conditional photon-number engineering changes the non-Gaussian state.",
    ),
    "08": (
        "Interference response",
        "The 33-point phase scan probes how the processed state responds to a Mach–Zehnder phase shift.",
    ),
    "09": (
        "Quadrature conditioning",
        "Homodyne selects one measured quadrature, leaving a conditional idler state whose phase-space structure can be inspected.",
    ),
    "10": (
        "Complex-amplitude conditioning",
        "Heterodyne samples both quadratures simultaneously, giving a two-dimensional measurement outcome.",
    ),
    "11": (
        "Readout comparison",
        "Compare homodyne and heterodyne conditioning to see how the measurement model changes the inferred state.",
    ),
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def run_timestamp() -> str:
    value = os.environ.get("REPORT_TIMESTAMP")
    if value:
        return value
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def refine_run(run_root: Path, timestamp: str) -> None:
    page = run_root / "index.html"
    text = page.read_text(encoding="utf-8")

    # The stage cards are the meaningful visual entry points; the old gallery
    # repeated those same images, so remove it and its navigation affordance.
    text = re.sub(r'\s*<a href="#gallery">plots</a>', "", text)
    text = re.sub(
        r'\s*<section class="section alt" id="gallery">.*?</section>',
        "",
        text,
        flags=re.S,
    )

    insight_html = (
        '<div class="stage-insight"><strong>__TITLE__</strong><span>__TEXT__</span></div>'
    )
    for number, (title, body) in INSIGHTS.items():
        pattern = rf'(<article class="stage"[^>]*id="stage-{re.escape(number)}".*?</div><div class="stage-body">.*?<p>.*?</p>)(</div></article>)'
        replacement = rf"\1{insight_html.replace('__TITLE__', esc(title)).replace('__TEXT__', esc(body))}\2"
        text = re.sub(pattern, replacement, text, count=1, flags=re.S)

    timestamp_label = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    text = text.replace(
        '<span class="badge">commit <code>',
        f'<span class="badge">{esc(timestamp_label)}</span><span class="badge">commit <code>',
        1,
    )
    text = text.replace(
        '<div class="footer">',
        '<div class="footer">',
    )
    page.write_text(text, encoding="utf-8")

    (run_root / "run_metadata.txt").write_text(
        f"timestamp={timestamp}\ncommit={os.environ.get('REPORT_COMMIT', os.environ.get('GITHUB_SHA', 'unknown'))}\n",
        encoding="utf-8",
    )


def read_timestamp(run_dir: Path) -> str:
    metadata = run_dir / "run_metadata.txt"
    if metadata.exists():
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.startswith("timestamp="):
                return line.removeprefix("timestamp=")
    return "legacy run"


def display_timestamp(value: str) -> str:
    if value == "legacy run":
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d · %H:%M UTC"
        )
    except ValueError:
        return value


def rebuild_archive(runs_root: Path) -> None:
    cards = []
    run_dirs = sorted(
        (p for p in runs_root.iterdir() if p.is_dir()),
        key=lambda p: read_timestamp(p),
        reverse=True,
    )
    for run in run_dirs:
        commit = run.name
        timestamp = display_timestamp(read_timestamp(run))
        cards.append(
            '<a class="run" href="runs/__RUN__/">'
            '<span class="dot"></span><span class="run-copy">'
            '<small class="run-time">__TIME__</small>'
            "<strong>__HASH__</strong>"
            '<span class="run-detail">complex simulation · open explorer →</span>'
            "</span></a>".replace("__RUN__", esc(commit))
            .replace("__TIME__", esc(timestamp))
            .replace("__HASH__", esc(commit[:12]))
        )

    index = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Catsy Lab · Simulation archive</title><style>:root{color-scheme:dark;--bg:#070b12;--panel:#111827;--line:#263244;--text:#edf2f7;--muted:#94a3b8;--accent:#67e8f9;--violet:#c4b5fd}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0%,#0d2940,transparent 34rem),radial-gradient(circle at 90% 20%,#211a3c,transparent 36rem),var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{width:min(1040px,calc(100% - 32px));margin:auto;padding:82px 0}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:800}h1{font-size:clamp(42px,7vw,72px);line-height:.98;letter-spacing:-.05em;margin:8px 0 18px}p{color:var(--muted);max-width:70ch}.archive{margin-top:38px;display:grid;gap:10px}.run{display:grid;grid-template-columns:12px 1fr;gap:15px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(17,24,39,.88),rgba(10,16,27,.8));padding:18px 20px;border-radius:16px;transition:.2s}.run:hover{border-color:rgba(103,232,249,.5);transform:translateX(4px);box-shadow:0 14px 45px rgba(0,0,0,.2)}.dot{width:10px;height:10px;margin-top:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 18px rgba(103,232,249,.8)}.run-copy{display:grid;gap:2px}.run-time{color:var(--accent);font:700 11px ui-monospace,monospace;letter-spacing:.04em}.run strong{font:800 16px ui-monospace,monospace}.run-detail{color:var(--muted);font-size:11px}.hint{margin-top:30px;padding:15px 17px;border:1px solid var(--line);border-radius:14px;background:rgba(15,23,42,.55);color:var(--muted);font-size:12px}</style></head><body><main><div class="eyebrow">Catsy · commit-addressed simulation archive</div><h1>Complex experiment runs</h1><p>Browse the visual history of the Gaussian → Fock → interferometric workflow. Each run is timestamped before its source hash, with the stage explorer and journal kept together for reproducibility.</p><div class="archive">__CARDS__</div><div class="hint">Tip: open a run to follow the state transformation stage by stage. Diagnostics are attached to the physical step they explain rather than duplicated in a separate gallery.</div></main></body></html>""".replace(
        "__CARDS__", "".join(cards) or "<p>No reports yet.</p>"
    )
    (OUTPUT_ROOT / "index.html").write_text(index, encoding="utf-8")


def main() -> None:
    run_root = (
        OUTPUT_ROOT
        / "runs"
        / os.environ.get("REPORT_COMMIT", os.environ.get("GITHUB_SHA", "unknown"))
    )
    if not (run_root / "index.html").exists():
        raise SystemExit(f"Missing generated report: {run_root / 'index.html'}")
    timestamp = run_timestamp()
    refine_run(run_root, timestamp)
    rebuild_archive(OUTPUT_ROOT / "runs")


if __name__ == "__main__":
    main()

"""Build a restrained, commit-addressed static report for the complex example."""

from __future__ import annotations

import html
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_ROOT = Path("_site")
RUN_ROOT = Path("runs/complex_circuit")

STAGES = [
    (
        "01",
        "Gaussian preparation",
        "gaussian",
        "Prepare and couple the three-mode Gaussian state.",
        "01_final_signal_phase_space",
        "Check the initial signal phase-space geometry before the non-Gaussian operations.",
        "Gaussian state / signal mode",
    ),
    (
        "02",
        "Covariance structure",
        "gaussian",
        "Inspect second-order moments of the final Gaussian circuit state.",
        "02_final_covariance_matrix",
        "Look for squeezing, variances, and cross-mode covariance structure.",
        "Covariance matrix",
    ),
    (
        "03",
        "Mode correlations",
        "gaussian",
        "Visualize correlations established between signal, idler, and reference.",
        "03_final_mode_correlations",
        "Use this as the correlation baseline for the later conditioned measurements.",
        "Inter-mode correlations",
    ),
    (
        "04",
        "Even cat preparation",
        "fock",
        "Prepare the non-Gaussian even Schrödinger cat state.",
        "04_even_cat_wigner",
        "Look for the characteristic interference structure distinguishing the cat from a Gaussian state.",
        "Wigner representation",
    ),
    (
        "05",
        "Cat-state diagnostics",
        "fock",
        "Inspect the prepared cat in the complementary state representations.",
        "05_even_cat_state",
        "Compare occupation structure and phase-space features before heralding.",
        "Fock / phase-space diagnostics",
    ),
    (
        "06",
        "Photon subtraction",
        "fock",
        "Apply realistic photon subtraction with finite tap reflectivity and detector efficiency.",
        "06_after_photon_subtraction",
        "Assess how the heralded operation changes the non-Gaussian state.",
        "Conditioned Fock state",
    ),
    (
        "07",
        "Photon addition",
        "fock",
        "Follow subtraction with realistic photon addition.",
        "07_after_photon_addition",
        "Compare the state with the subtraction output and track the effect of the second heralded operation.",
        "Conditioned Fock state",
    ),
    (
        "08",
        "Mach–Zehnder scan",
        "interferometer",
        "Probe the processed state through a lossy 33-point phase scan.",
        None,
        "The journal records the phase-dependent interferometric response; this stage is the bridge from state preparation to readout.",
        "33 phase points",
    ),
    (
        "09",
        "Homodyne readout",
        "measurement",
        "Condition the Gaussian state on a signal quadrature measurement.",
        "08_after_homodyne_idler",
        "Inspect the conditioned idler state and the effect of selecting one quadrature outcome.",
        "Quadrature measurement",
    ),
    (
        "10",
        "Heterodyne readout",
        "measurement",
        "Condition the Gaussian state on simultaneous x/p detection.",
        "09_after_heterodyne_idler",
        "Compare the conditioned state with homodyne, noting the different measurement information retained.",
        "Phase-space measurement",
    ),
    (
        "11",
        "Measurement comparison",
        "measurement",
        "Compare the states produced by the two measurement schemes.",
        "10_measurement_conditioning",
        "Use this final diagnostic to connect the measurement outcomes with their conditioned states.",
        "Homodyne vs heterodyne",
    ),
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_run_page(site_run_root: Path, commit: str, generated_at: str) -> None:
    short_commit = commit[:12]
    plots = sorted((site_run_root / "plots").glob("*.png"))
    by_stem = {plot.stem: plot for plot in plots}
    journals = sorted(
        path
        for path in site_run_root.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"}
    )

    pipeline = []
    stages = []
    for number, title, category, description, plot_stem, inspect, result in STAGES:
        pipeline.append(
            f'<a class="pipeline-step" href="#stage-{esc(number)}">'
            f"<span>{esc(number)}</span><strong>{esc(title)}</strong><small>{esc(category)}</small></a>"
        )
        plot = by_stem.get(plot_stem) if plot_stem else None
        image = ""
        if plot:
            relative = esc(plot.relative_to(site_run_root).as_posix())
            visual = (
                f'<button class="stage-plot" data-src="{relative}" data-title="{esc(title)}" type="button">'
                f'<img src="{relative}" alt="{esc(title)}" loading="lazy"></button>'
            )
        else:
            visual = '<div class="stage-plot empty"><span>—</span><small>response recorded in journal</small></div>'
        stages.append(
            f'<article class="stage" data-category="{esc(category)}" id="stage-{esc(number)}">'
            f'<div class="stage-number">{esc(number)}</div>{visual}'
            f'<div class="stage-copy"><div class="kicker">{esc(category)}</div>'
            f"<h3>{esc(title)}</h3><p>{esc(description)}</p>"
            f'<div class="meaning"><strong>What to inspect</strong><span>{esc(inspect)}</span></div>'
            f'<div class="result"><strong>Diagnostic</strong><span>{esc(result)}</span></div>'
            f"</div></article>"
        )

    journal_links = []
    for journal in journals:
        relative = esc(journal.relative_to(site_run_root).as_posix())
        journal_links.append(
            f'<a class="journal-file" href="{relative}" target="_blank" rel="noopener">'
            f'<span class="file-type">{esc(journal.suffix[1:].upper())}</span>'
            f"<span><strong>{relative}</strong><small>{journal.stat().st_size:,} bytes · open raw file ↗</small></span></a>"
        )

    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Catsy complex simulation report for __SHORT_COMMIT__">
<title>Catsy · Complex simulation · __SHORT_COMMIT__</title>
<style>
:root{color-scheme:dark;--bg:#0a0d12;--panel:#10151d;--panel2:#141a23;--line:#252d39;--text:#e7ebf0;--muted:#8994a3;--accent:#8fb8c5;--accent-soft:rgba(143,184,197,.10)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.65 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}a{color:inherit;text-decoration:none}button{font:inherit;color:inherit}.container{width:min(1120px,calc(100% - 40px));margin:auto}
.topbar{position:sticky;top:0;z-index:20;background:rgba(10,13,18,.94);border-bottom:1px solid var(--line)}.topbar-inner{min-height:54px;display:flex;align-items:center;justify-content:space-between}.brand{font-weight:750;letter-spacing:.02em}.brand span{color:var(--accent)}.nav{display:flex;gap:18px;color:var(--muted);font-size:12px}.nav a:hover{color:var(--text)}
.hero{padding:64px 0 46px;border-bottom:1px solid var(--line)}.eyebrow,.kicker{color:var(--accent);text-transform:uppercase;letter-spacing:.14em;font-size:10px;font-weight:750}.hero h1{max-width:850px;margin:8px 0 14px;font-size:clamp(36px,5vw,60px);line-height:1.02;letter-spacing:-.045em;font-weight:700}.lead{max-width:780px;color:#b5bec9;font-size:16px}.meta{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}.badge,.button,.filter{border:1px solid var(--line);background:var(--panel);border-radius:7px;padding:6px 10px;font-size:11px;color:#b8c1cc}.badge strong{color:var(--text)}.button{display:inline-flex}.button:hover,.filter:hover{border-color:#485564;color:var(--text)}.links{display:flex;gap:8px;flex-wrap:wrap}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:24px}.metric{padding:14px 15px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.metric .value{font-size:22px;font-weight:700}.metric .label{margin-top:2px;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.1em}
.section{padding:42px 0}.section.alt{background:#0d1117;border-block:1px solid var(--line)}.section-head{margin-bottom:20px}.section-head h2{margin:3px 0 4px;font-size:26px;letter-spacing:-.025em}.section-head p{margin:0;color:var(--muted)}
.pipeline{display:grid;grid-template-columns:repeat(11,1fr);gap:6px}.pipeline-step{min-height:88px;padding:11px 9px;border:1px solid var(--line);border-radius:8px;background:var(--panel);transition:border-color .15s}.pipeline-step:hover{border-color:#4a5968}.pipeline-step span{font:700 10px ui-monospace,monospace;color:var(--accent)}.pipeline-step strong{display:block;margin-top:7px;font-size:10px;line-height:1.3}.pipeline-step small{display:block;margin-top:5px;color:var(--muted);font-size:9px}
.filters{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px}.filter{cursor:pointer}.filter.active{background:var(--accent-soft);border-color:#506b76;color:#dce8eb}.stages{display:grid;gap:12px}.stage{display:grid;grid-template-columns:38px minmax(220px,38%) 1fr;min-height:220px;border:1px solid var(--line);border-radius:10px;background:var(--panel);overflow:hidden}.stage-number{padding:16px 9px;color:#657181;font:700 11px ui-monospace,monospace}.stage-plot{width:100%;min-height:220px;border:0;padding:0;background:#080b10;cursor:zoom-in;overflow:hidden}.stage-plot img{display:block;width:100%;height:100%;min-height:220px;object-fit:cover}.stage-plot.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;color:#52606f;font-size:26px}.stage-plot.empty small{font-size:10px;color:var(--muted)}.stage-copy{padding:18px 20px}.stage-copy h3{margin:4px 0 7px;font-size:18px}.stage-copy p{margin:0 0 14px;color:var(--muted);font-size:13px}.meaning,.result{display:grid;grid-template-columns:100px 1fr;gap:8px;padding:9px 0;border-top:1px solid var(--line);font-size:11px}.meaning strong,.result strong{color:#b7c0cb;font-weight:650}.meaning span,.result span{color:var(--muted)}
.journal{display:grid;gap:7px;max-width:820px}.journal-file{display:flex;align-items:center;gap:12px;padding:11px 13px;border:1px solid var(--line);border-radius:8px;background:var(--panel)}.journal-file:hover{border-color:#485564}.file-type{width:36px;height:30px;display:grid;place-items:center;border-radius:5px;background:var(--accent-soft);color:var(--accent);font:700 9px ui-monospace,monospace}.journal-file strong,.journal-file small{display:block}.journal-file strong{font-size:11px}.journal-file small{color:var(--muted);font-size:10px}
.modal{position:fixed;inset:0;z-index:50;display:none;place-items:center;padding:24px;background:rgba(0,0,0,.86)}.modal.open{display:grid}.modal-inner{width:min(1180px,96vw);max-height:94vh}.modal-head,.modal-foot{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:8px 0}.modal-title{font-weight:650}.modal-close{border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:5px 9px;cursor:pointer}.modal img{display:block;width:100%;max-height:80vh;object-fit:contain;background:#05070a;border:1px solid var(--line)}.modal-foot{color:var(--muted);font-size:10px}
.footer{padding:30px 0 55px;border-top:1px solid var(--line);color:var(--muted);font-size:11px}@media(max-width:950px){.pipeline{grid-template-columns:repeat(4,1fr)}.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.container{width:calc(100% - 24px)}.nav{display:none}.metrics{grid-template-columns:1fr 1fr}.stage{grid-template-columns:30px 1fr}.stage-plot{grid-column:2;min-height:190px}.stage-copy{grid-column:2}.pipeline{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header class="topbar"><div class="container topbar-inner"><a class="brand" href="../..">catsy <span>LAB</span></a><nav class="nav"><a href="../..">runs</a><a href="#pipeline">pipeline</a><a href="#stages">stages</a><a href="#journal">journal</a></nav></div></header>
<main><section class="hero"><div class="container"><div class="eyebrow">Complex simulation · __GENERATED_AT__</div><h1>Gaussian preparation, non-Gaussian processing, interferometry and readout.</h1><p class="lead">A compact visual record of the three-mode experiment. Each diagnostic is kept with the physical stage it describes, with enough context to interpret the result without turning the report into a plot catalogue.</p><div class="meta"><span class="badge">commit <strong>__SHORT_COMMIT__</strong></span><span class="badge">__PLOT_COUNT__ diagnostics</span><span class="badge">__JOURNAL_COUNT__ journal files</span></div><div class="links"><a class="button" href="https://github.com/raiyiz/catsy/commit/__COMMIT__" target="_blank" rel="noopener">source commit ↗</a><a class="button" href="https://github.com/raiyiz/catsy/actions" target="_blank" rel="noopener">CI ↗</a><a class="button" href="../..">all runs</a></div><div class="metrics"><div class="metric"><div class="value">3</div><div class="label">Gaussian modes</div></div><div class="metric"><div class="value">2</div><div class="label">readout schemes</div></div><div class="metric"><div class="value">33</div><div class="label">MZI phase points</div></div><div class="metric"><div class="value">__PLOT_COUNT__</div><div class="label">saved diagnostics</div></div></div></div></section>
<section class="section alt" id="pipeline"><div class="container"><div class="section-head"><div class="eyebrow">Experiment map</div><h2>From state preparation to measurement</h2><p>Each step links to the diagnostic and its physical interpretation below.</p></div><div class="pipeline">__PIPELINE__</div></div></section>
<section class="section" id="stages"><div class="container"><div class="section-head"><div class="eyebrow">Stage diagnostics</div><h2>What happened at each step</h2><p>Plots are shown once, beside the operation they document.</p></div><div class="filters"><button class="filter active" data-filter="all" type="button">all</button><button class="filter" data-filter="gaussian" type="button">Gaussian</button><button class="filter" data-filter="fock" type="button">Fock</button><button class="filter" data-filter="interferometer" type="button">interferometer</button><button class="filter" data-filter="measurement" type="button">measurement</button></div><div class="stages">__STAGES__</div></div></section>
<section class="section alt" id="journal"><div class="container"><div class="section-head"><div class="eyebrow">Reproducibility</div><h2>Experiment journal</h2><p>Machine-readable records remain beside the visual diagnostics.</p></div><div class="journal">__JOURNAL__</div></div></section></main>
<div class="modal" id="viewer" role="dialog" aria-modal="true" aria-label="Plot viewer"><div class="modal-inner"><div class="modal-head"><span class="modal-title" id="viewer-title"></span><button class="modal-close" id="viewer-close" type="button">close</button></div><img id="viewer-image" alt=""><div class="modal-foot"><span>Esc to close · click outside to close</span><a id="viewer-open" href="#" target="_blank" rel="noopener">open original ↗</a></div></div></div>
<footer class="footer"><div class="container">Catsy · complex simulation · commit <code>__COMMIT__</code><br>Static report generated by CI. Visualizations are produced through Catsy's plotting helpers.</div></footer>
<script>
(function(){const modal=document.getElementById('viewer'),img=document.getElementById('viewer-image'),title=document.getElementById('viewer-title'),open=document.getElementById('viewer-open');function close(){modal.classList.remove('open');img.src='';document.body.style.overflow='';}document.querySelectorAll('[data-src]').forEach(function(el){el.addEventListener('click',function(){img.src=el.dataset.src;img.alt=el.dataset.title;title.textContent=el.dataset.title;open.href=el.dataset.src;modal.classList.add('open');document.body.style.overflow='hidden';});});document.getElementById('viewer-close').addEventListener('click',close);modal.addEventListener('click',function(e){if(e.target===modal)close();});document.addEventListener('keydown',function(e){if(e.key==='Escape')close();});document.querySelectorAll('.filter').forEach(function(btn){btn.addEventListener('click',function(){document.querySelectorAll('.filter').forEach(function(b){b.classList.remove('active');});btn.classList.add('active');const f=btn.dataset.filter;document.querySelectorAll('.stage').forEach(function(s){s.style.display=f==='all'||s.dataset.category===f?'grid':'none';});});});})();
</script></body></html>"""

    report = (
        template.replace("__SHORT_COMMIT__", esc(short_commit))
        .replace("__COMMIT__", esc(commit))
        .replace("__GENERATED_AT__", esc(generated_at))
        .replace("__PLOT_COUNT__", str(len(plots)))
        .replace("__JOURNAL_COUNT__", str(len(journals)))
        .replace("__PIPELINE__", "".join(pipeline))
        .replace("__STAGES__", "".join(stages))
        .replace(
            "__JOURNAL__",
            "".join(journal_links) or "<p>No journal files were generated.</p>",
        )
    )
    (site_run_root / "index.html").write_text(report, encoding="utf-8")


def main() -> None:
    if not RUN_ROOT.exists():
        raise SystemExit(f"Missing complex-example output: {RUN_ROOT}")

    commit = os.environ.get("GITHUB_SHA", "unknown")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d · %H:%M UTC")
    site_run_root = OUTPUT_ROOT / "runs" / commit
    site_run_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_ROOT, site_run_root, dirs_exist_ok=True)
    build_run_page(site_run_root, commit, generated_at)

    runs_root = OUTPUT_ROOT / "runs"
    run_dirs = (
        sorted(
            (p for p in runs_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        if runs_root.exists()
        else []
    )
    cards = "".join(
        f'<a class="run" href="runs/{esc(p.name)}/"><span class="dot"></span>'
        f"<span><strong>{esc(run_datetime(p))}</strong><small>{esc(p.name[:12])} · open simulation explorer →</small></span></a>"
        for p in run_dirs
    )
    index = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Catsy · Simulation archive</title><style>
:root{color-scheme:dark;--bg:#0a0d12;--panel:#10151d;--line:#252d39;--text:#e7ebf0;--muted:#8994a3;--accent:#8fb8c5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 Inter,ui-sans-serif,system-ui,sans-serif}main{width:min(920px,calc(100% - 32px));margin:auto;padding:72px 0}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.14em;font-size:10px;font-weight:750}h1{font-size:clamp(38px,6vw,58px);line-height:1;letter-spacing:-.04em;margin:8px 0 14px}p{max-width:70ch;color:var(--muted)}.run{display:flex;align-items:center;gap:13px;padding:14px;border:1px solid var(--line);background:var(--panel);border-radius:8px;margin:7px 0}.run:hover{border-color:#485564}.run strong,.run small{display:block}.run strong{font-size:12px}.run small{color:var(--muted);font:10px ui-monospace,monospace;margin-top:2px}.dot{width:7px;height:7px;border-radius:50%;background:var(--accent);flex:none}</style></head><body><main><div class="eyebrow">Catsy · commit-addressed simulation archive</div><h1>Complex experiment runs</h1><p>Visual history of the Gaussian → Fock → interferometric workflow. Each run keeps its stage diagnostics and machine-readable journal together.</p><div style="margin-top:30px">__CARDS__</div></main></body></html>""".replace(
        "__CARDS__", cards or "<p>No reports yet.</p>"
    )
    (OUTPUT_ROOT / "index.html").write_text(index, encoding="utf-8")


def run_datetime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime(
            "%Y-%m-%d · %H:%M UTC"
        )
    except OSError:
        return "time unavailable"


if __name__ == "__main__":
    main()

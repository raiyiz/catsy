"""Build a commit-addressed, visual static HTML report for the complex example."""

from __future__ import annotations

import html
import os
import shutil
from pathlib import Path

OUTPUT_ROOT = Path("_site")
RUN_ROOT = Path("runs/complex_circuit")

STAGES = [
    (
        "01",
        "Gaussian preparation",
        "Prepare and couple the three-mode Gaussian state.",
        "01_final_signal_phase_space",
    ),
    (
        "02",
        "Gaussian diagnostics",
        "Inspect covariance and inter-mode correlations.",
        "02_final_covariance_matrix",
    ),
    (
        "03",
        "Mode correlations",
        "Visualize the correlations created by the Gaussian circuit.",
        "03_final_mode_correlations",
    ),
    (
        "04",
        "Even cat state",
        "Prepare the non-Gaussian even Schrödinger cat.",
        "04_even_cat_wigner",
    ),
    (
        "05",
        "Cat-state diagnostics",
        "Inspect the cat in the Fock basis and phase space.",
        "05_even_cat_state",
    ),
    (
        "06",
        "Heralded photon subtraction",
        "Apply a lossy, detector-limited subtraction event.",
        "06_after_photon_subtraction",
    ),
    (
        "07",
        "Heralded photon addition",
        "Follow subtraction with realistic photon addition.",
        "07_after_photon_addition",
    ),
    (
        "08",
        "Mach–Zehnder interferometer",
        "Probe the processed state across a lossy phase scan.",
        None,
    ),
    (
        "09",
        "Homodyne readout",
        "Condition the Gaussian state on a signal quadrature measurement.",
        "08_after_homodyne_idler",
    ),
    (
        "10",
        "Heterodyne readout",
        "Condition the Gaussian state on simultaneous x/p detection.",
        "09_after_heterodyne_idler",
    ),
    (
        "11",
        "Measurement comparison",
        "Compare the conditioned phase-space states.",
        "10_measurement_conditioning",
    ),
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def plot_map(plots: list[Path]) -> dict[str, Path]:
    return {plot.stem: plot for plot in plots}


def build_run_page(site_run_root: Path, commit: str) -> None:
    short_commit = commit[:12]
    plots = sorted((site_run_root / "plots").glob("*.png"))
    by_stem = {plot.stem: plot for plot in plots}
    journals = sorted(
        path for path in site_run_root.rglob("*") if path.suffix.lower() in {".json", ".jsonl"}
    )

    gallery = []
    for plot in plots:
        relative = plot.relative_to(site_run_root).as_posix()
        title = plot.stem.replace("_", " ").title()
        gallery.append(
            '<button class="gallery-card" data-src="__SRC__" data-title="__TITLE__" type="button">'
            '<span class="gallery-image"><img src="__SRC__" alt="__TITLE__" loading="lazy"></span>'
            '<span class="gallery-caption"><strong>__TITLE__</strong><small>inspect ↗</small></span></button>'
            .replace("__SRC__", esc(relative))
            .replace("__TITLE__", esc(title))
        )

    stage_cards = []
    pipeline_steps = []
    for number, title, description, plot_stem, category in STAGES:
        pipeline_steps.append(
            '<a class="pipeline-step" href="#stage-__N__"><span class="num">__N__</span>'
            '<strong>__TITLE__</strong><small>__CATEGORY__</small></a>'
            .replace("__N__", esc(number))
            .replace("__TITLE__", esc(title))
            .replace("__CATEGORY__", esc(category))
        )

        plot = by_stem.get(plot_stem) if plot_stem else None
        image = ""
        if plot:
            relative = plot.relative_to(site_run_root).as_posix()
            visual = (
                '<button class="stage-image" data-src="__SRC__" data-title="__TITLE__" type="button">'
                '<img src="__SRC__" alt="__TITLE__" loading="lazy"></button>'
                .replace("__SRC__", esc(relative))
                .replace("__TITLE__", esc(title))
            )
        elif "Mach" in title:
            image = '<div class="stage-image empty"><span>◌</span><small>phase scan recorded in journal</small></div>'
        else:
            visual = '<div class="stage-image empty"><span>∿</span><small>journal-only stage</small></div>'

        stage_cards.append(
            '<article class="stage" data-category="__CATEGORY__" id="stage-__N__">'
            '<div class="stage-index">__N__</div>__VISUAL__'
            '<div class="stage-body"><div class="stage-kicker">__CATEGORY__</div>'
            '<h3>__TITLE__</h3><p>__DESCRIPTION__</p></div></article>'
            .replace("__CATEGORY__", esc(category))
            .replace("__N__", esc(number))
            .replace("__VISUAL__", visual)
            .replace("__TITLE__", esc(title))
            .replace("__DESCRIPTION__", esc(description))
        )

    journal_links = []
    for journal in journals:
        relative = journal.relative_to(site_run_root).as_posix()
        size = journal.stat().st_size
        journal_links.append(
            '<a class="journal-file" href="__PATH__" target="_blank" rel="noopener">'
            '<span class="file-icon">__EXT__</span><span><strong>__PATH__</strong>'
            '<small>__SIZE__ bytes · open raw file ↗</small></span></a>'
            .replace("__PATH__", esc(relative))
            .replace("__EXT__", esc(journal.suffix[1:].upper()))
            .replace("__SIZE__", f"{journal.stat().st_size:,}")
        )

    plot_count = len(plots)
    journal_count = len(journals)
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Interactive Catsy complex experiment report for __SHORT_COMMIT__">
<title>Catsy Lab · Complex experiment · __SHORT_COMMIT__</title>
<style>
:root { color-scheme:dark; --bg:#070b12; --panel:#101722; --panel2:#151e2c; --line:rgba(148,163,184,.17); --text:#edf2f7; --muted:#93a4b8; --signal:#67e8f9; --violet:#c4b5fd; --pink:#f0a0bd; --gold:#f4c66d; --green:#86efac; }
*{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:radial-gradient(circle at 10% 0%,rgba(34,211,238,.12),transparent 34rem),radial-gradient(circle at 90% 15%,rgba(139,92,246,.13),transparent 36rem),var(--bg);color:var(--text);font:15px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif} body::before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:36px 36px;mask-image:linear-gradient(to bottom,black,transparent 80%)} a{color:inherit;text-decoration:none} button{font:inherit;color:inherit} .container{width:min(1240px,calc(100% - 40px));margin:auto}
.topbar{position:sticky;top:0;z-index:30;backdrop-filter:blur(18px);background:rgba(7,11,18,.76);border-bottom:1px solid var(--line)} .topbar-inner{min-height:58px;display:flex;align-items:center;justify-content:space-between;gap:20px} .brand{font-weight:900;letter-spacing:.02em} .brand span{color:var(--signal)} .nav{display:flex;gap:18px;color:var(--muted);font-size:13px} .nav a:hover{color:var(--text)}
.hero{padding:78px 0 54px} .eyebrow,.stage-kicker{color:var(--signal);text-transform:uppercase;letter-spacing:.16em;font-size:10px;font-weight:800} h1{margin:9px 0 15px;font-size:clamp(40px,6vw,72px);line-height:.96;letter-spacing:-.055em;max-width:900px} .lead{max-width:820px;color:#cbd5e1;font-size:18px} .badges,.links,.filters{display:flex;flex-wrap:wrap;gap:8px} .badges{margin:25px 0 20px} .badge,.button,.filter{border:1px solid var(--line);background:rgba(15,23,42,.66);border-radius:999px;padding:6px 11px;font-size:12px;color:#cbd5e1} .badge.good{color:var(--green)} .button{border-radius:10px;display:inline-flex} .button:hover,.filter:hover{border-color:rgba(103,232,249,.5);color:var(--signal)}
.meta-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:25px} .metric{padding:17px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(145deg,rgba(20,31,46,.82),rgba(11,17,27,.8));box-shadow:0 18px 60px rgba(0,0,0,.16)} .metric .value{font-size:25px;font-weight:850;letter-spacing:-.04em} .metric .label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.13em}
.section{padding:48px 0} .section.alt{background:rgba(15,23,42,.35);border-top:1px solid var(--line);border-bottom:1px solid var(--line)} .section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px} h2{margin:0;font-size:30px;letter-spacing:-.035em} .section-head p{margin:5px 0 0;color:var(--muted)}
.pipeline{display:grid;grid-template-columns:repeat(11,1fr);gap:7px} .pipeline-step{position:relative;min-height:108px;padding:13px 10px;border:1px solid var(--line);border-radius:14px;background:linear-gradient(145deg,rgba(30,41,59,.75),rgba(15,23,42,.54));transition:.2s} .pipeline-step:hover{transform:translateY(-2px);border-color:rgba(103,232,249,.45)} .pipeline-step::after{content:"→";position:absolute;right:-11px;top:40%;color:#64748b;z-index:2} .pipeline-step:last-child::after{display:none} .pipeline-step .num{color:var(--signal);font:800 10px ui-monospace,monospace} .pipeline-step strong{display:block;margin-top:8px;font-size:11px;line-height:1.25} .pipeline-step small{display:block;color:var(--muted);margin-top:6px;font-size:9px;line-height:1.3}
.filters{margin-bottom:16px} .filter{cursor:pointer} .filter.active{background:var(--signal);color:#061016;border-color:var(--signal);font-weight:800} .stages{display:grid;grid-template-columns:repeat(2,1fr);gap:16px} .stage{display:grid;grid-template-columns:42px minmax(180px,42%) 1fr;min-height:214px;overflow:hidden;border:1px solid var(--line);border-radius:18px;background:linear-gradient(145deg,rgba(17,25,38,.9),rgba(11,17,27,.82));transition:transform .2s,border-color .2s,opacity .2s} .stage:hover{transform:translateY(-2px);border-color:rgba(103,232,249,.35)} .stage.hidden{display:none} .stage-index{padding:18px 9px;color:var(--violet);font:800 11px ui-monospace,monospace} .stage-image{border:0;padding:0;width:100%;min-height:214px;background:#05080e;overflow:hidden;cursor:zoom-in} .stage-image img{width:100%;height:100%;min-height:214px;object-fit:cover;display:block;transition:transform .35s} .stage-image:hover img{transform:scale(1.025)} .stage-image.empty{display:flex;align-items:center;justify-content:center;flex-direction:column;color:#64748b;font-size:38px} .stage-image.empty small{font-size:10px;color:var(--muted)} .stage-body{padding:21px 20px} .stage-body h3{margin:6px 0 8px;font-size:18px;letter-spacing:-.02em} .stage-body p{margin:0;color:var(--muted);font-size:13px}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:15px} .gallery-card{padding:0;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:var(--panel);cursor:zoom-in;text-align:left;transition:.2s} .gallery-card:hover{transform:translateY(-3px);border-color:rgba(196,181,253,.5)} .gallery-image{display:block;aspect-ratio:16/10;background:#05080e} .gallery-image img{width:100%;height:100%;object-fit:contain;display:block} .gallery-caption{display:flex;justify-content:space-between;gap:10px;padding:12px 14px} .gallery-caption strong{font-size:12px} .gallery-caption small{color:var(--muted);font-size:10px}
.journal{display:grid;gap:10px;max-width:900px} .journal-file{display:flex;align-items:center;gap:14px;border:1px solid var(--line);border-radius:13px;padding:13px;background:var(--panel)} .journal-file:hover{border-color:rgba(103,232,249,.4)} .file-icon{width:42px;height:42px;display:grid;place-items:center;border-radius:10px;background:rgba(103,232,249,.08);color:var(--signal);font:800 10px ui-monospace,monospace} .journal-file strong,.journal-file small{display:block} .journal-file strong{font-size:13px} .journal-file small{color:var(--muted);font-size:11px}
.modal{position:fixed;inset:0;z-index:100;display:none;place-items:center;padding:24px;background:rgba(0,0,0,.82);backdrop-filter:blur(12px)} .modal.open{display:grid} .modal-inner{width:min(1200px,96vw);max-height:94vh;display:grid;grid-template-rows:auto 1fr auto;gap:10px} .modal-head,.modal-foot{display:flex;align-items:center;justify-content:space-between;gap:15px} .modal-title{font-weight:800} .modal-close{border:1px solid var(--line);background:var(--panel);border-radius:9px;padding:7px 12px;cursor:pointer} .modal img{width:100%;max-height:78vh;object-fit:contain;background:#03060b;border:1px solid var(--line);border-radius:12px} .modal-foot{color:var(--muted);font-size:11px}
.footer{padding:35px 0 70px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
@media(max-width:1000px){.meta-grid{grid-template-columns:repeat(2,1fr)}.pipeline{grid-template-columns:repeat(4,1fr)}.pipeline-step::after{display:none}.stages,.gallery{grid-template-columns:1fr}} @media(max-width:620px){.container{width:min(100% - 24px,1240px)}.hero{padding-top:52px}.nav{display:none}.meta-grid{grid-template-columns:1fr 1fr}.stage{grid-template-columns:34px 1fr}.stage-image{grid-column:2;min-height:180px}.stage-body{grid-column:2}.pipeline{grid-template-columns:repeat(2,1fr)}}
</style></head>
<body>
<header class="topbar"><div class="container topbar-inner"><a class="brand" href="../..">catsy <span>LAB</span></a><nav class="nav"><a href="../../">all runs</a><a href="#pipeline">pipeline</a><a href="#gallery">plots</a><a href="#journal">journal</a></nav></div></header>
<main>
<section class="hero"><div class="container"><div class="eyebrow">Commit-addressed simulation explorer</div><h1>Gaussian → non-Gaussian → interferometric readout</h1><p class="lead">A visual laboratory notebook for the Catsy complex example: three-mode Gaussian preparation, heralded Fock-space processing, a lossy Mach–Zehnder scan, and complementary homodyne / heterodyne measurements.</p>
<div class="badges"><span class="badge good">● CI generated</span><span class="badge">commit <code>__SHORT_COMMIT__</code></span><span class="badge">__PLOT_COUNT__ diagnostics</span><span class="badge">__JOURNAL_COUNT__ journal files</span></div>
<div class="meta-grid"><div class="metric"><div class="value">3</div><div class="label">Gaussian modes</div></div><div class="metric"><div class="value">2</div><div class="label">measurement schemes</div></div><div class="metric"><div class="value">33</div><div class="label">MZI phase points</div></div><div class="metric"><div class="value">__PLOT_COUNT__</div><div class="label">saved diagnostics</div></div></div>
<div class="links" style="margin-top:20px"><a class="button" href="https://github.com/raiyiz/catsy/commit/__COMMIT__" target="_blank" rel="noopener">view commit ↗</a><a class="button" href="https://github.com/raiyiz/catsy/actions" target="_blank" rel="noopener">view CI ↗</a><a class="button" href="../../">← all runs</a></div></div></section>
<section class="section alt" id="pipeline"><div class="container"><div class="section-head"><div><div class="eyebrow">The calculation</div><h2>One experiment, eleven moments</h2><p>Follow the state as it crosses representations, operations, interferometry, and measurement.</p></div></div><div class="pipeline">__PIPELINE__</div></div></section>
<section class="section" id="stages"><div class="container"><div class="section-head"><div><div class="eyebrow">Simulation explorer</div><h2>Explore the experiment</h2><p>Filter the physical layer you care about, then open any diagnostic at full resolution.</p></div></div><div class="filters" role="toolbar" aria-label="Filter experiment stages"><button class="filter active" data-filter="all" type="button">all stages</button><button class="filter" data-filter="gaussian" type="button">Gaussian</button><button class="filter" data-filter="fock" type="button">Fock space</button><button class="filter" data-filter="interferometer" type="button">interferometer</button><button class="filter" data-filter="measurement" type="button">measurements</button></div><div class="stages">__STAGES__</div></div></section>
<section class="section alt" id="gallery"><div class="container"><div class="section-head"><div><div class="eyebrow">Diagnostics</div><h2>Visual gallery</h2><p>Click a plot to inspect it in the built-in viewer. No download required.</p></div></div><div class="gallery">__GALLERY__</div></div></section>
<section class="section" id="journal"><div class="container"><div class="section-head"><div><div class="eyebrow">Reproducibility</div><h2>Experiment journal</h2><p>The raw journal lives beside the plots so the visual record and machine-readable record stay together.</p></div></div><div class="journal">__JOURNAL__</div></div></section>
</main>
<div class="modal" id="viewer" role="dialog" aria-modal="true" aria-label="Plot viewer"><div class="modal-inner"><div class="modal-head"><span class="modal-title" id="viewer-title"></span><button class="modal-close" id="viewer-close" type="button">close ×</button></div><img id="viewer-image" alt=""><div class="modal-foot"><span>Esc to close · click outside to close</span><a id="viewer-open" href="#" target="_blank" rel="noopener">open original ↗</a></div></div></div>
<footer class="footer"><div class="container"><strong>Catsy Lab</strong> · complex example · commit <code>__COMMIT__</code><br><span>Static report generated by CI; visualizations are produced exclusively through Catsy's plotting helpers.</span></div></footer>
<script>
(function(){
  const viewer=document.getElementById('viewer'), image=document.getElementById('viewer-image'), title=document.getElementById('viewer-title'), original=document.getElementById('viewer-open');
  function openViewer(src,label){ image.src=src; image.alt=label; title.textContent=label; original.href=src; viewer.classList.add('open'); document.body.style.overflow='hidden'; document.getElementById('viewer-close').focus(); }
  function closeViewer(){ viewer.classList.remove('open'); image.src=''; document.body.style.overflow=''; }
  document.querySelectorAll('[data-src]').forEach(function(el){ el.addEventListener('click',function(){ openViewer(el.dataset.src,el.dataset.title); }); });
  document.getElementById('viewer-close').addEventListener('click',closeViewer);
  viewer.addEventListener('click',function(e){ if(e.target===viewer) closeViewer(); });
  document.addEventListener('keydown',function(e){ if(e.key==='Escape') closeViewer(); });
  document.querySelectorAll('.filter').forEach(function(button){ button.addEventListener('click',function(){ document.querySelectorAll('.filter').forEach(function(b){b.classList.remove('active');}); button.classList.add('active'); const filter=button.dataset.filter; document.querySelectorAll('.stage').forEach(function(stage){ stage.classList.toggle('hidden',filter!=='all' && stage.dataset.category!==filter); }); }); });
})();
</script></body></html>"""

    report = (
        template.replace("__SHORT_COMMIT__", esc(short_commit))
        .replace("__COMMIT__", esc(commit))
        .replace("__PLOT_COUNT__", str(plot_count))
        .replace("__JOURNAL_COUNT__", str(journal_count))
        .replace("__PIPELINE__", "".join(pipeline_steps))
        .replace("__STAGES__", "".join(stage_cards))
        .replace("__GALLERY__", "".join(gallery) or "<p>No plots were generated.</p>")
        .replace("__JOURNAL__", "".join(journal_links) or "<p>No journal files were generated.</p>")
    )
    (site_run_root / "index.html").write_text(report, encoding="utf-8")


def main() -> None:
    if not RUN_ROOT.exists():
        raise SystemExit(f"Missing complex-example output: {RUN_ROOT}")

    commit = os.environ.get("GITHUB_SHA", "unknown")
    site_run_root = OUTPUT_ROOT / "runs" / commit
    site_run_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUN_ROOT, site_run_root, dirs_exist_ok=True)
    build_run_page(site_run_root, commit)

    runs_root = OUTPUT_ROOT / "runs"
    run_dirs = (
        sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        if runs_root.exists()
        else []
    )
    cards = "".join(
        '<a class="run" href="runs/__RUN__/">'
        '<span class="dot"></span><span><strong>__RUN_SHORT__</strong>'
        '<small>open simulation explorer →</small></span></a>'
        .replace("__RUN__", esc(p.name))
        .replace("__RUN_SHORT__", esc(p.name[:12]))
        for p in run_dirs
    )
    index = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Catsy Lab · Simulation archive</title><style>:root{color-scheme:dark;--bg:#070b12;--panel:#111827;--line:#263244;--text:#edf2f7;--muted:#94a3b8;--accent:#67e8f9}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0%,#0d2940,transparent 34rem),radial-gradient(circle at 90% 20%,#211a3c,transparent 36rem),var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{width:min(960px,calc(100% - 32px));margin:auto;padding:90px 0}.eyebrow{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:800}h1{font-size:clamp(42px,7vw,72px);line-height:.98;letter-spacing:-.05em;margin:8px 0 18px}p{color:var(--muted);max-width:70ch}.run{display:flex;align-items:center;gap:15px;border:1px solid var(--line);background:rgba(17,24,39,.78);padding:17px;border-radius:15px;margin:10px 0;transition:.2s}.run:hover{border-color:#4b637c;transform:translateX(3px)}.run strong,.run small{display:block}.run strong{font:800 14px ui-monospace,monospace}.run small{color:var(--muted);font-size:11px}.dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 18px rgba(103,232,249,.8)}</style></head><body><main><div class="eyebrow">Catsy · commit-addressed simulation archive</div><h1>Complex experiment runs</h1><p>Browse the visual history of the Gaussian → Fock → interferometric workflow. Every run keeps its plots and journal output together and is tied to the exact source commit that produced it.</p><div style="margin-top:34px">__CARDS__</div></main></body></html>"""
        .replace("__CARDS__", cards or "<p>No reports yet.</p>")
    )
    (OUTPUT_ROOT / "index.html").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()

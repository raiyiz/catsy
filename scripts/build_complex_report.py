"""Build a commit-addressed, visual static HTML report for the complex example."""

from __future__ import annotations

import html
import os
from pathlib import Path

OUTPUT_ROOT = Path("_site")
RUN_ROOT = Path("runs/complex_circuit")

STAGES = [
    ("01", "Gaussian preparation", "Prepare and couple the three-mode Gaussian state.", "01_final_signal_phase_space"),
    ("02", "Gaussian diagnostics", "Inspect covariance and inter-mode correlations.", "02_final_covariance_matrix"),
    ("03", "Mode correlations", "Visualize the correlations created by the Gaussian circuit.", "03_final_mode_correlations"),
    ("04", "Even cat state", "Prepare the non-Gaussian even Schrödinger cat.", "04_even_cat_wigner"),
    ("05", "Cat-state diagnostics", "Inspect the cat in the Fock basis and phase space.", "05_even_cat_state"),
    ("06", "Heralded photon subtraction", "Apply a lossy, detector-limited subtraction event.", "06_after_photon_subtraction"),
    ("07", "Heralded photon addition", "Follow subtraction with realistic photon addition.", "07_after_photon_addition"),
    ("08", "Mach–Zehnder interferometer", "Probe the processed state across a lossy phase scan.", None),
    ("09", "Homodyne readout", "Condition the Gaussian state on a signal quadrature measurement.", "08_after_homodyne_idler"),
    ("10", "Heterodyne readout", "Condition the Gaussian state on simultaneous x/p detection.", "09_after_heterodyne_idler"),
    ("11", "Measurement comparison", "Compare the conditioned phase-space states.", "10_measurement_conditioning"),
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def plot_map(plots: list[Path]) -> dict[str, Path]:
    return {plot.stem: plot for plot in plots}


def build_run_page(site_run_root: Path, commit: str) -> None:
    short_commit = commit[:12]
    plots = sorted((site_run_root / "plots").glob("*.png"))
    by_stem = plot_map(plots)
    journals = sorted(
        path
        for path in site_run_root.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"}
    )

    gallery = []
    for plot in plots:
        relative = plot.relative_to(site_run_root).as_posix()
        title = plot.stem.replace("_", " ").title()
        gallery.append(
            f'''<a class="gallery-card" href="{esc(relative)}" target="_blank">'''
            f'''<div class="gallery-image"><img src="{esc(relative)}" alt="{esc(title)}" loading="lazy"></div>'''
            f'''<div class="gallery-caption"><span>{esc(title)}</span><small>open full resolution ↗</small></div></a>'''
        )

    stage_cards = []
    for number, title, description, plot_stem in STAGES:
        plot = by_stem.get(plot_stem) if plot_stem else None
        image = ""
        if plot:
            relative = plot.relative_to(site_run_root).as_posix()
            image = (
                f'<a class="stage-image" href="{esc(relative)}" target="_blank">'
                f'<img src="{esc(relative)}" alt="{esc(title)}" loading="lazy"></a>'
            )
        elif "Mach" in title:
            image = '<div class="stage-image empty"><span>◌</span><small>phase scan recorded in journal</small></div>'
        else:
            image = '<div class="stage-image empty"><span>∿</span><small>journal-only stage</small></div>'
        stage_cards.append(
            f'''<article class="stage">'''
            f'''<div class="stage-index">{esc(number)}</div>'''
            f'''{image}<div class="stage-body"><h3>{esc(title)}</h3><p>{esc(description)}</p></div></article>'''
        )

    journal_links = []
    for journal in journals:
        relative = journal.relative_to(site_run_root).as_posix()
        size = journal.stat().st_size
        journal_links.append(
            f'''<a class="journal-file" href="{esc(relative)}" target="_blank">'''
            f'''<span class="file-icon">{journal.suffix[1:].upper()}</span>'''
            f'''<span><strong>{esc(relative)}</strong><small>{size:,} bytes · open raw file ↗</small></span></a>'''
        )

    plot_count = len(plots)
    journal_count = len(journals)
    report = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Commit-addressed Catsy complex experiment report for {esc(short_commit)}">
<title>Catsy Lab · Complex experiment · {esc(short_commit)}</title>
<style>
:root {{ color-scheme: dark; --bg:#070b12; --panel:rgba(17,24,39,.78); --line:rgba(148,163,184,.16); --text:#edf2f7; --muted:#94a3b8; --accent:#7dd3fc; --accent2:#c4b5fd; --good:#86efac; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:radial-gradient(circle at 15% 0%,rgba(56,189,248,.13),transparent 30rem),radial-gradient(circle at 90% 15%,rgba(139,92,246,.13),transparent 32rem),var(--bg); color:var(--text); font:15px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
a {{ color:inherit; text-decoration:none; }}
.container {{ width:min(1240px,calc(100% - 40px)); margin:auto; }}
.topbar {{ position:sticky; top:0; z-index:10; backdrop-filter:blur(18px); background:rgba(7,11,18,.72); border-bottom:1px solid var(--line); }}
.topbar-inner {{ min-height:58px; display:flex; align-items:center; justify-content:space-between; gap:20px; }}
.brand {{ font-weight:800; letter-spacing:.02em; }}
.brand span {{ color:var(--accent); }}
.nav {{ display:flex; gap:18px; color:var(--muted); font-size:13px; }}
.nav a:hover {{ color:var(--text); }}
.hero {{ padding:78px 0 52px; position:relative; }}
.eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.18em; font-size:11px; font-weight:800; }}
h1 {{ margin:10px 0 14px; font-size:clamp(38px,6vw,70px); line-height:.98; letter-spacing:-.045em; max-width:850px; }}
.lead {{ max-width:800px; color:#cbd5e1; font-size:18px; }}
.badges {{ display:flex; flex-wrap:wrap; gap:8px; margin:25px 0; }}
.badge {{ border:1px solid var(--line); background:rgba(15,23,42,.62); border-radius:999px; padding:5px 10px; color:#cbd5e1; font-size:12px; }}
.badge.good {{ color:var(--good); }}
.meta-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:30px; }}
.metric {{ padding:17px; border:1px solid var(--line); border-radius:16px; background:var(--panel); box-shadow:0 18px 50px rgba(0,0,0,.15); }}
.metric .value {{ font-size:24px; font-weight:800; letter-spacing:-.03em; }}
.metric .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em; margin-top:2px; }}
.section {{ padding:45px 0; }}
.section-head {{ display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:22px; }}
h2 {{ margin:0; font-size:30px; letter-spacing:-.03em; }}
.section-head p {{ margin:5px 0 0; color:var(--muted); }}
.pipeline {{ display:grid; grid-template-columns:repeat(11,1fr); gap:8px; align-items:stretch; }}
.pipeline-step {{ position:relative; min-height:104px; padding:13px 10px; border:1px solid var(--line); border-radius:14px; background:linear-gradient(145deg,rgba(30,41,59,.72),rgba(15,23,42,.55)); }}
.pipeline-step::after {{ content:"→"; position:absolute; right:-12px; top:40%; color:#64748b; z-index:2; }}
.pipeline-step:last-child::after {{ display:none; }}
.pipeline-step .num {{ color:var(--accent); font:700 11px ui-monospace,SFMono-Regular,monospace; }}
.pipeline-step strong {{ display:block; margin-top:8px; font-size:12px; line-height:1.25; }}
.pipeline-step small {{ display:block; color:var(--muted); margin-top:6px; font-size:10px; line-height:1.25; }}
.stages {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px; }}
.stage {{ display:grid; grid-template-columns:44px minmax(180px,42%) 1fr; min-height:205px; overflow:hidden; border:1px solid var(--line); border-radius:18px; background:var(--panel); transition:transform .2s,border-color .2s; }}
.stage:hover {{ transform:translateY(-2px); border-color:rgba(125,211,252,.35); }}
.stage-index {{ padding:18px 10px; color:var(--accent2); font:800 12px ui-monospace,SFMono-Regular,monospace; }}
.stage-image {{ display:block; min-height:205px; background:#05080e; overflow:hidden; }}
.stage-image img {{ width:100%; height:100%; min-height:205px; object-fit:cover; display:block; }}
.stage-image.empty {{ display:flex; align-items:center; justify-content:center; flex-direction:column; color:#64748b; font-size:38px; }}
.stage-image.empty small {{ font-size:10px; color:var(--muted); }}
.stage-body {{ padding:22px 20px; }}
.stage-body h3 {{ margin:0 0 8px; font-size:18px; }}
.stage-body p {{ margin:0; color:var(--muted); font-size:13px; }}
.gallery {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
.gallery-card {{ border:1px solid var(--line); border-radius:16px; overflow:hidden; background:var(--panel); transition:transform .2s,border-color .2s; }}
.gallery-card:hover {{ transform:translateY(-3px); border-color:rgba(196,181,253,.45); }}
.gallery-image {{ aspect-ratio:16/10; background:#05080e; }}
.gallery-image img {{ width:100%; height:100%; object-fit:contain; display:block; }}
.gallery-caption {{ display:flex; justify-content:space-between; gap:10px; padding:12px 14px; }}
.gallery-caption span {{ font-weight:700; font-size:13px; }}
.gallery-caption small {{ color:var(--muted); font-size:10px; white-space:nowrap; }}
.journal {{ display:grid; gap:10px; max-width:850px; }}
.journal-file {{ display:flex; align-items:center; gap:14px; border:1px solid var(--line); border-radius:13px; padding:13px; background:var(--panel); }}
.journal-file:hover {{ border-color:rgba(125,211,252,.35); }}
.file-icon {{ width:42px; height:42px; display:grid; place-items:center; border-radius:10px; background:rgba(125,211,252,.1); color:var(--accent); font:800 10px ui-monospace,monospace; }}
.journal-file strong,.journal-file small {{ display:block; }}
.journal-file strong {{ font-size:13px; }}
.journal-file small {{ color:var(--muted); font-size:11px; }}
.footer {{ padding:35px 0 70px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); }}
.links {{ display:flex; gap:12px; flex-wrap:wrap; }}
.button {{ border:1px solid var(--line); padding:8px 12px; border-radius:10px; background:rgba(15,23,42,.65); }}
.button:hover {{ border-color:var(--accent); color:var(--accent); }}
@media(max-width:900px) {{ .meta-grid {{ grid-template-columns:repeat(2,1fr); }} .pipeline {{ grid-template-columns:repeat(3,1fr); }} .pipeline-step::after {{ display:none; }} .stages,.gallery {{ grid-template-columns:1fr; }} }}
@media(max-width:600px) {{ .container {{ width:min(100% - 24px,1240px); }} .hero {{ padding-top:52px; }} .meta-grid {{ grid-template-columns:1fr 1fr; }} .stage {{ grid-template-columns:34px 1fr; }} .stage-image {{ grid-column:2; min-height:180px; }} .stage-body {{ grid-column:2; }} .nav {{ display:none; }} }}
</style>
</head>
<body>
<header class="topbar"><div class="container topbar-inner"><a class="brand" href="../..">catsy <span>LAB</span></a><nav class="nav"><a href="../../">all runs</a><a href="#pipeline">pipeline</a><a href="#gallery">plots</a><a href="#journal">journal</a></nav></div></header>
<main>
<section class="hero"><div class="container">
<div class="eyebrow">Commit-addressed experiment report</div>
<h1>Gaussian → non-Gaussian → interferometric readout</h1>
<p class="lead">A visual record of the Catsy complex example: three-mode Gaussian state preparation, heralded Fock-space processing, lossy Mach–Zehnder interferometry, and complementary homodyne / heterodyne measurements.</p>
<div class="badges"><span class="badge good">● CI generated</span><span class="badge">commit <code>{esc(short_commit)}</code></span><span class="badge">{plot_count} plots</span><span class="badge">{journal_count} journal files</span></div>
<div class="meta-grid">
<div class="metric"><div class="value">3</div><div class="label">Gaussian modes</div></div>
<div class="metric"><div class="value">2</div><div class="label">Measurement schemes</div></div>
<div class="metric"><div class="value">33</div><div class="label">MZI phase points</div></div>
<div class="metric"><div class="value">{plot_count}</div><div class="label">Saved diagnostics</div></div>
</div>
<div class="links" style="margin-top:20px"><a class="button" href="https://github.com/raiyiz/catsy/commit/{esc(commit)}" target="_blank">view commit ↗</a><a class="button" href="https://github.com/raiyiz/catsy/actions" target="_blank">view CI ↗</a><a class="button" href="../../">← all runs</a></div>
</div></section>

<section class="section" id="pipeline"><div class="container"><div class="section-head"><div><h2>The experiment pipeline</h2><p>From Gaussian preparation to two independent signal readouts.</p></div></div>
<div class="pipeline">{''.join(f'<div class="pipeline-step"><span class="num">{esc(n)}</span><strong>{esc(t)}</strong><small>{esc(d)}</small></div>' for n,t,d,_ in STAGES)}</div>
</div></section>

<section class="section"><div class="container"><div class="section-head"><div><h2>Stage-by-stage</h2><p>Each card links directly to the diagnostic produced at that point in the experiment.</p></div></div>
<div class="stages">{''.join(stage_cards)}</div></div></section>

<section class="section" id="gallery"><div class="container"><div class="section-head"><div><h2>Visual gallery</h2><p>Full-resolution Catsy diagnostics. Click any image to inspect it without leaving the report.</p></div></div>
<div class="gallery">{''.join(gallery) or '<p>No plots were generated.</p>'}</div></div></section>

<section class="section" id="journal"><div class="container"><div class="section-head"><div><h2>Experiment journal</h2><p>Raw journal output is preserved beside the plots for reproducibility.</p></div></div>
<div class="journal">{''.join(journal_links) or '<p>No journal files were generated.</p>'}</div></div></section>
</main>
<footer class="footer"><div class="container"><strong>Catsy Lab</strong> · complex example · commit <code>{esc(commit)}</code><br><span>Static report generated by CI; plots are produced exclusively through Catsy's plotting helpers.</span></div></footer>
</body></html>'''
    (site_run_root / "index.html").write_text(report, encoding="utf-8")


def main() -> None:
    if not RUN_ROOT.exists():
        raise SystemExit(f"Missing complex-example output: {RUN_ROOT}")

    commit = os.environ.get("GITHUB_SHA", "unknown")
    site_run_root = OUTPUT_ROOT / "runs" / commit
    site_run_root.mkdir(parents=True, exist_ok=True)

    import shutil

    shutil.copytree(RUN_ROOT, site_run_root, dirs_exist_ok=True)
    build_run_page(site_run_root, commit)

    runs_root = OUTPUT_ROOT / "runs"
    run_dirs = (
        sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True)
        if runs_root.exists()
        else []
    )
    cards = []
    for path in run_dirs:
        cards.append(
            f'''<a class="run" href="runs/{esc(path.name)}/"><span class="dot"></span><span><strong>{esc(path.name[:12])}</strong><small>open experiment report →</small></span></a>'''
        )
    index = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Catsy Lab · Complex experiment archive</title>
<style>:root{{color-scheme:dark;--bg:#070b12;--panel:#111827;--line:#263244;--text:#edf2f7;--muted:#94a3b8;--accent:#7dd3fc}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 20% 0%,#0d2940,transparent 35rem),radial-gradient(circle at 90% 20%,#211a3c,transparent 35rem),var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}}main{{width:min(900px,calc(100% - 32px));margin:0 auto;padding:90px 0}}h1{{font-size:clamp(40px,7vw,68px);line-height:1;letter-spacing:-.05em;margin:8px 0 18px}}p{{color:var(--muted)}}.eyebrow{{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:11px;font-weight:800}}.run{{display:flex;align-items:center;gap:15px;border:1px solid var(--line);background:rgba(17,24,39,.75);padding:17px;border-radius:15px;margin:10px 0;transition:.2s}}.run:hover{{border-color:#4b637c;transform:translateX(3px)}}.run strong,.run small{{display:block}}.run strong{{font:800 14px ui-monospace,monospace}}.run small{{color:var(--muted);font-size:11px}}.dot{{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 18px rgba(125,211,252,.8)}}code{{color:#c4b5fd}}</style></head>
<body><main><div class="eyebrow">Catsy · CI experiment archive</div><h1>Complex experiment runs</h1><p>Commit-addressed visual reports. Every run preserves its plots and journal output together, making the generated experiment history browsable and reproducible.</p><div style="margin-top:34px">{''.join(cards) or '<p>No reports yet.</p>'}</div></main></body></html>'''
    (OUTPUT_ROOT / "index.html").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Build the Pages site: one section per example script, each keeping its
own history of runs. Reads REPORT_COMMIT/REPORT_REF/REPORT_RUN_ID and the
platform-specific REPORT_COMMIT_URL/REPORT_CI_URL from the environment, so
the same script produces correct links on both GitHub and GitLab without
any post-generation text patching.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

OUTPUT_ROOT = Path("_site")
REPO_ROOT = Path(__file__).resolve().parent.parent


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _read_run_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        data = tomllib.load(file)
    run = data.get("run", {})
    return run if isinstance(run, dict) else {}


# ---------------------------------------------------------------------------
# Example registry — add an entry here (plus a matching CI job) to give a
# new example script its own Pages section.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleSpec:
    id: str
    title: str
    tagline: str
    run_root: Path
    config_path: Path
    kind: str  # "rich" (stage diagnostics + plots) | "journal_only"


EXAMPLES: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        id="complex-example",
        title="Complex simulation",
        tagline=(
            "Gaussian preparation, non-Gaussian processing, interferometry, "
            "and dual readout."
        ),
        run_root=Path("runs/complex_circuit"),
        config_path=REPO_ROOT / "examples" / "config.toml",
        kind="rich",
    ),
    ExampleSpec(
        id="complex-circuit",
        title="Three-mode circuit",
        tagline=(
            "The minimal three-mode Gaussian circuit and journal-persistence "
            "example from the docs."
        ),
        run_root=Path("runs/complex_circuit_basic"),
        config_path=REPO_ROOT / "examples" / "config_circuit.toml",
        kind="journal_only",
    ),
)


# ---------------------------------------------------------------------------
# Stage diagnostics (the "rich" example only)
# ---------------------------------------------------------------------------

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

INSIGHTS = {
    "01": "The phase-space ellipse is the state's uncertainty region before anything non-Gaussian happens to it.",
    "02": "Read the diagonal as single-mode variances and the off-diagonal blocks as inter-mode covariance.",
    "03": "Correlated quadratures here are the Gaussian coupling the later conditioned measurements will draw on.",
    "04": "A two-lobed Wigner function with interference fringes is the signature a Gaussian description cannot produce.",
    "05": "Occupation structure in Fock space is the complementary view of the same non-Gaussian state.",
    "06": "Finite tap reflectivity and detector efficiency make this a realistic conditional operation, not an ideal one.",
    "07": "Compare directly against subtraction to see how heralded photon-number engineering reshapes the state.",
    "08": "The phase scan is the bridge from state preparation to a readout that depends on interferometer phase.",
    "09": "Homodyne selects one quadrature; the conditioned idler state reflects only that partial information.",
    "10": "Heterodyne samples both quadratures at once, trading precision for a genuinely two-dimensional outcome.",
    "11": "Same input state, two measurement models — the difference is what each scheme lets you infer.",
}

CATEGORY_LABELS = {
    "gaussian": "Gaussian",
    "fock": "Fock",
    "interferometer": "Interferometer",
    "measurement": "Measurement",
}


# ---------------------------------------------------------------------------
# Circuit diagram — shared by both examples, since build_circuit() in
# complex_example.py and complex_circuit.py is the same eight-operation
# sequence. r/eta come from each example's own config so the diagram can
# never drift out of sync with the numbers it's showing.
# ---------------------------------------------------------------------------

CIRCUIT_ROWS = ["signal", "idler", "reference"]
CIRCUIT_LAST_COL = 9


def circuit_ops_for(config: dict[str, object]) -> list[tuple]:
    r = float(config.get("signal_squeezing", 0.6))  # type: ignore[arg-type]
    eta = float(config.get("signal_idler_transmissivity", 0.65))  # type: ignore[arg-type]
    return [
        (1, "box", "signal", "Squeeze", f"r = {r:.2f}"),
        (2, "box", "signal", "Displace", "α"),
        (3, "link", ("signal", "idler"), "Beam splitter", f"η = {eta:.2f}"),
        (4, "box", "idler", "Rotate", "φ = 0.35"),
        (5, "box", "idler", "Thermal loss", "η = 0.90, n̄ = 0.15"),
        (6, "box", "reference", "Squeeze", "r = 0.35, θ = π/4"),
        (7, "link", ("idler", "reference"), "Beam splitter", "η = 0.50"),
        (8, "box", "signal", "Loss", "η = 0.92"),
    ]


def render_circuit_diagram(ops: list[tuple], *, with_readout: bool) -> str:
    col_w, x0 = 108, 60
    row_y = {"signal": 56, "idler": 156, "reference": 256}
    box_w, box_h = 96, 46

    def x_of(col: int) -> float:
        return x0 + col * col_w

    height = 320
    parts: list[str] = []

    for row in CIRCUIT_ROWS:
        y = row_y[row]
        parts.append(
            f'<line class="c-wire" x1="{x0 - 20}" y1="{y}" x2="{x_of(CIRCUIT_LAST_COL) + 60}" y2="{y}"></line>'
        )
        parts.append(
            f'<text class="c-row-label" x="{x0 - 20}" y="{y - 16}">{esc(row)}</text>'
        )
        parts.append(f'<text class="c-vac" x="{x0 - 20}" y="{y + 4}">|0⟩</text>')

    for col, kind, row, label, sub in ops:
        cx = x_of(col)
        if kind == "box":
            y = row_y[row]
            parts.append(
                f'<rect class="c-box" x="{cx - box_w / 2:.1f}" y="{y - box_h / 2:.1f}" '
                f'width="{box_w}" height="{box_h}" rx="8"></rect>'
            )
            parts.append(
                f'<text class="c-label" x="{cx}" y="{y - 3}" text-anchor="middle">{esc(label)}</text>'
            )
            parts.append(
                f'<text class="c-sub" x="{cx}" y="{y + 13}" text-anchor="middle">{esc(sub)}</text>'
            )
        else:
            row_a, row_b = row
            y_a, y_b = row_y[row_a], row_y[row_b]
            parts.append(
                f'<line class="c-wire c-link" x1="{cx}" y1="{y_a}" x2="{cx}" y2="{y_b}"></line>'
            )
            mid_y = (y_a + y_b) / 2
            parts.append(
                f'<rect class="c-box" x="{cx - box_w / 2:.1f}" y="{mid_y - box_h / 2:.1f}" '
                f'width="{box_w}" height="{box_h}" rx="8"></rect>'
            )
            parts.append(
                f'<text class="c-label" x="{cx}" y="{mid_y - 3}" text-anchor="middle">{esc(label)}</text>'
            )
            parts.append(
                f'<text class="c-sub" x="{cx}" y="{mid_y + 13}" text-anchor="middle">{esc(sub)}</text>'
            )

    if with_readout:
        for row in ("idler", "reference"):
            y = row_y[row]
            end_x = x_of(CIRCUIT_LAST_COL) - 10
            parts.append(
                f'<text class="c-port" x="{end_x}" y="{y + 4}" text-anchor="end">→ readout</text>'
            )

        sig_y = row_y["signal"]
        fork_x = x_of(8) + box_w / 2 + 26
        for dy, label, sub in ((-52, "Homodyne", "φ = π/6"), (52, "Heterodyne", "x, p")):
            end_y = sig_y + dy
            parts.append(
                f'<path class="c-wire" d="M {fork_x - 26} {sig_y} '
                f'C {fork_x + 10} {sig_y}, {fork_x + 10} {end_y}, {fork_x + 46} {end_y}" fill="none"></path>'
            )
            parts.append(
                f'<rect class="c-box meas" x="{fork_x + 46:.1f}" y="{end_y - box_h / 2:.1f}" '
                f'width="{box_w}" height="{box_h}" rx="8"></rect>'
            )
            parts.append(
                f'<text class="c-label" x="{fork_x + 46 + box_w / 2}" y="{end_y - 3}" text-anchor="middle">{esc(label)}</text>'
            )
            parts.append(
                f'<text class="c-sub" x="{fork_x + 46 + box_w / 2}" y="{end_y + 13}" text-anchor="middle">{esc(sub)}</text>'
            )
        width = fork_x + 46 + box_w + 30
    else:
        for row in CIRCUIT_ROWS:
            y = row_y[row]
            end_x = x_of(CIRCUIT_LAST_COL) - 10
            parts.append(
                f'<text class="c-port" x="{end_x}" y="{y + 4}" text-anchor="end">→ journal</text>'
            )
        width = x_of(CIRCUIT_LAST_COL) + 90

    return (
        f'<svg class="circuit" viewBox="0 0 {width:.0f} {height}" role="img" '
        f'aria-label="Signal, idler, and reference mode wires through the Gaussian circuit.">'
        + "".join(parts)
        + "</svg>"
    )


# ---------------------------------------------------------------------------
# Shared page chrome
# ---------------------------------------------------------------------------

FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&'
    'family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">'
)

PAGE_STYLE = """
:root{
  color-scheme:dark;
  --bg:#14161c; --surface:#191d26; --surface-2:#1f232e; --line:#2a303e;
  --text:#e7e9f0; --muted:#8a91a6;
  --gaussian:#5fd3e8; --fock:#f0568e; --interferometer:#e8b34c; --measurement:#b3a6f2;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);
  font:14px/1.65 'IBM Plex Sans',ui-sans-serif,system-ui,-apple-system,sans-serif}
a{color:inherit;text-decoration:none}
button{font:inherit;color:inherit}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace}
.container{width:min(1120px,calc(100% - 40px));margin:auto}
h1,h2,h3{font-family:'Fraunces',serif;font-weight:600;letter-spacing:-0.02em;margin:0}

.topbar{position:sticky;top:0;z-index:20;background:rgba(20,22,28,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.topbar-inner{min-height:56px;display:flex;align-items:center;justify-content:space-between;gap:14px}
.brand{font-family:'Fraunces',serif;font-weight:600;font-size:16px;display:flex;align-items:center;gap:8px}
.brand .dot{width:8px;height:8px;border-radius:50%;background:var(--gaussian)}
.crumbs{color:var(--muted);font-size:12.5px;display:flex;align-items:center;gap:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.crumbs a:hover{color:var(--text)}
.crumbs .sep{opacity:.5}
.nav{display:flex;gap:20px;color:var(--muted);font-size:12.5px;flex:none}
.nav a:hover{color:var(--text)}

.hero{padding:60px 0 44px;border-bottom:1px solid var(--line);
  background:radial-gradient(480px 240px at 88% -10%,rgba(95,211,232,.05),transparent 60%)}
.eyebrow{color:var(--gaussian);text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:600;
  font-family:'IBM Plex Mono',monospace}
.hero h1{max-width:820px;margin:10px 0 14px;font-size:clamp(30px,4.4vw,46px);line-height:1.08}
.lead{max-width:72ch;color:#b7bfcb;font-size:15.5px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0 0}
.badge{border:1px solid var(--line);background:var(--surface);border-radius:7px;padding:6px 11px;font-size:12px;color:#b8c1cc}
.badge strong{color:var(--text)}
.links{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
.button{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);background:var(--surface);
  border-radius:7px;padding:8px 13px;font-size:12.5px;transition:border-color .15s,color .15s}
.button:hover{border-color:#3a4257;color:var(--text)}

.section{padding:48px 0}
.section.alt{background:var(--surface);border-block:1px solid var(--line)}
.section-head{margin-bottom:22px;max-width:60ch}
.section-head h2{font-size:24px}
.section-head p{margin:8px 0 0;color:var(--muted);font-size:14.5px}

.circuit-card{background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:26px 24px 20px;overflow-x:auto}
.section.alt .circuit-card{background:var(--surface-2)}
svg.circuit{width:100%;min-width:640px;height:auto;display:block}
.c-wire{stroke:var(--line);stroke-width:2}
.c-link{stroke:var(--gaussian);stroke-width:2;opacity:.5}
.c-box{fill:rgba(95,211,232,.10);stroke:var(--gaussian);stroke-width:1.4}
.c-box.meas{fill:rgba(179,166,242,.14);stroke:var(--measurement)}
.c-label{font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:600;fill:var(--text)}
.c-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;fill:var(--muted)}
.c-row-label{font-family:'IBM Plex Mono',monospace;font-size:11px;fill:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.c-vac{font-family:'IBM Plex Mono',monospace;font-size:11px;fill:var(--muted)}
.c-port{font-family:'IBM Plex Mono',monospace;font-size:10px;fill:var(--muted)}
.circuit-legend{display:flex;gap:18px;margin-top:14px;flex-wrap:wrap}
.circuit-legend span{display:inline-flex;align-items:center;gap:7px;font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.circuit-legend .sw{width:10px;height:10px;border-radius:3px;display:inline-block}

.pipeline{display:grid;grid-template-columns:repeat(11,1fr);gap:6px}
.pipeline-step{min-height:92px;padding:12px 10px;border:1px solid var(--line);border-radius:9px;
  background:var(--bg);border-left:3px solid var(--step-color);transition:border-color .15s,transform .15s}
.section.alt .pipeline-step{background:var(--surface-2)}
.pipeline-step:hover{transform:translateY(-2px)}
.pipeline-step .n{font:700 10px 'IBM Plex Mono',monospace;color:var(--step-color)}
.pipeline-step strong{display:block;margin-top:7px;font-size:10.5px;line-height:1.3;font-family:'IBM Plex Sans',sans-serif;font-weight:600}
.pipeline-step small{display:block;margin-top:5px;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.06em;font-family:'IBM Plex Mono',monospace}

.filters{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:16px}
.filter{cursor:pointer;border:1px solid var(--line);background:var(--surface);border-radius:20px;padding:7px 14px;font-size:12px;color:var(--muted);transition:all .15s}
.filter.active{color:var(--bg);background:var(--text);border-color:var(--text);font-weight:600}
.section.alt .filter{background:var(--surface-2)}

.stages{display:grid;gap:14px}
.stage{display:grid;grid-template-columns:42px minmax(220px,36%) 1fr;border:1px solid var(--line);
  border-radius:12px;background:var(--surface);overflow:hidden}
.section.alt .stage{background:var(--surface-2)}
.stage-index{padding:16px 10px;display:flex;flex-direction:column;align-items:center;gap:10px;
  background:linear-gradient(180deg,var(--stage-color) 0%,transparent 2px)}
.stage-index .num{font:700 12px 'IBM Plex Mono',monospace;color:var(--stage-color)}
.stage-index .cat{writing-mode:vertical-rl;text-orientation:mixed;font-size:9.5px;color:var(--muted);
  letter-spacing:.1em;text-transform:uppercase}
.stage-plot{width:100%;min-height:220px;border:0;padding:0;background:#101318;cursor:zoom-in;overflow:hidden;border-inline:1px solid var(--line)}
.stage-plot img{display:block;width:100%;height:100%;min-height:220px;object-fit:cover;transition:transform .3s}
.stage-plot:hover img{transform:scale(1.03)}
.stage-plot.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;color:#52606f;font-size:22px;gap:6px}
.stage-plot.empty small{font-size:10px;color:var(--muted);text-align:center;padding:0 16px}
.stage-copy{padding:18px 22px}
.stage-copy h3{font-size:17px;margin-bottom:6px}
.stage-copy p{margin:0 0 14px;color:var(--muted);font-size:13px}
.callout{display:grid;grid-template-columns:96px 1fr;gap:10px;padding:9px 0;border-top:1px solid var(--line);font-size:11.5px}
.callout .lbl{color:#b7c0cb;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:10.5px;text-transform:uppercase;letter-spacing:.06em}
.callout span:last-child{color:var(--muted)}
.insight{margin-top:2px;padding:11px 13px;border-radius:8px;background:color-mix(in srgb, var(--stage-color) 12%, transparent);
  border:1px solid color-mix(in srgb, var(--stage-color) 30%, transparent);font-size:12px;color:var(--text)}
.insight strong{color:var(--stage-color);font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:.08em;display:block;margin-bottom:4px}

.metrics-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;max-width:820px}
.metric{padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:var(--bg)}
.section.alt .metric{background:var(--surface-2)}
.metric .value{font-family:'Fraunces',serif;font-size:22px;font-weight:600}
.metric .label{margin-top:3px;color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;font-family:'IBM Plex Mono',monospace}

.journal{display:grid;gap:8px;max-width:820px}
.journal-file{display:flex;align-items:center;gap:13px;padding:12px 14px;border:1px solid var(--line);border-radius:9px;background:var(--bg)}
.section.alt .journal-file{background:var(--surface-2)}
.journal-file:hover{border-color:#3a4257}
.file-type{width:38px;height:32px;display:grid;place-items:center;border-radius:6px;
  background:rgba(95,211,232,.12);color:var(--gaussian);font:700 9.5px 'IBM Plex Mono',monospace}
.journal-file strong,.journal-file small{display:block}
.journal-file strong{font-size:12px;font-family:'IBM Plex Mono',monospace}
.journal-file small{color:var(--muted);font-size:10.5px;margin-top:2px}

.modal{position:fixed;inset:0;z-index:50;display:none;place-items:center;padding:24px;background:rgba(14,16,21,.9);backdrop-filter:blur(3px)}
.modal.open{display:grid}
.modal-inner{width:min(1180px,96vw);max-height:94vh}
.modal-head,.modal-foot{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:8px 0}
.modal-title{font-weight:600;font-family:'Fraunces',serif}
.modal-close{border:1px solid var(--line);background:var(--surface);border-radius:6px;padding:6px 10px;cursor:pointer}
.modal img{display:block;width:100%;max-height:80vh;object-fit:contain;background:#0c0e12;border:1px solid var(--line);border-radius:6px}
.modal-foot{color:var(--muted);font-size:10.5px}

.footer{padding:32px 0 56px;border-top:1px solid var(--line);color:var(--muted);font-size:11.5px}

@media(max-width:950px){.pipeline{grid-template-columns:repeat(4,1fr)}}
@media(max-width:680px){.container{width:calc(100% - 24px)}.nav{display:none}
  .stage{grid-template-columns:30px 1fr}
  .stage-index .cat{display:none}
  .stage-plot{grid-column:2;min-height:190px}
  .stage-copy{grid-column:2}
  .pipeline{grid-template-columns:repeat(2,1fr)}}

/* archive / index pages */
.archive-list{margin-top:30px;display:grid;gap:10px}
.run-card{display:grid;grid-template-columns:12px 1fr;gap:15px;border:1px solid var(--line);
  background:linear-gradient(135deg,rgba(17,24,39,.5),rgba(10,16,27,.35));
  padding:18px 20px;border-radius:14px;transition:.2s}
.run-card:hover{border-color:rgba(95,211,232,.5);transform:translateX(4px)}
.run-dot{width:10px;height:10px;margin-top:6px;border-radius:50%;background:var(--gaussian)}
.run-copy{display:grid;gap:2px}
.run-time{color:var(--gaussian);font:600 11px 'IBM Plex Mono',monospace;letter-spacing:.03em}
.run-card strong{font:600 16px 'IBM Plex Mono',monospace}
.run-detail{color:var(--muted);font-size:11.5px}
.empty-state{margin-top:24px;padding:20px 22px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);font-size:13px}

.example-grid{margin-top:34px;display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.example-card{display:block;border:1px solid var(--line);border-radius:16px;padding:24px 22px;
  background:linear-gradient(160deg,rgba(17,24,39,.6),rgba(10,16,27,.4));transition:.2s}
.example-card:hover{border-color:rgba(95,211,232,.5);transform:translateY(-3px)}
.example-card .kind{font:600 10px 'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.1em;color:var(--gaussian)}
.example-card h2{margin-top:8px;font-size:21px}
.example-card p{margin-top:8px;color:var(--muted);font-size:13px}
.example-card .stat-row{margin-top:16px;display:flex;gap:16px;font-size:11.5px;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.example-card .stat-row strong{color:var(--text)}
"""

RUN_SCRIPT = """
(function(){
  const modal=document.getElementById('viewer');
  if(!modal) return;
  const img=document.getElementById('viewer-image'),title=document.getElementById('viewer-title'),
        openLink=document.getElementById('viewer-open');
  function close(){modal.classList.remove('open');img.src='';document.body.style.overflow='';}
  document.querySelectorAll('[data-src]').forEach(function(el){
    el.addEventListener('click',function(){
      img.src=el.dataset.src;img.alt=el.dataset.title;title.textContent=el.dataset.title;
      openLink.href=el.dataset.src;modal.classList.add('open');document.body.style.overflow='hidden';
    });
  });
  const closeBtn=document.getElementById('viewer-close');
  if(closeBtn) closeBtn.addEventListener('click',close);
  modal.addEventListener('click',function(e){if(e.target===modal)close();});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')close();});
  document.querySelectorAll('.filter').forEach(function(btn){
    btn.addEventListener('click',function(){
      document.querySelectorAll('.filter').forEach(function(b){b.classList.remove('active');});
      btn.classList.add('active');
      const f=btn.dataset.filter;
      document.querySelectorAll('.stage').forEach(function(s){
        s.style.display=(f==='all'||s.dataset.category===f)?'grid':'none';
      });
    });
  });
})();
"""


def _topbar(crumbs: list[tuple[str, str]], nav_links: list[tuple[str, str]]) -> str:
    """crumbs: list of (label, href); last crumb has no href (current page)."""
    crumb_html = []
    for i, (label, href) in enumerate(crumbs):
        if i:
            crumb_html.append('<span class="sep">/</span>')
        crumb_html.append(
            f'<a href="{esc(href)}">{esc(label)}</a>'
            if href
            else f"<span>{esc(label)}</span>"
        )
    nav_html = "".join(
        f'<a href="{esc(href)}">{esc(label)}</a>' for label, href in nav_links
    )
    return (
        '<header class="topbar"><div class="container topbar-inner">'
        '<div class="crumbs">' + "".join(crumb_html) + "</div>"
        f'<nav class="nav">{nav_html}</nav>'
        "</div></header>"
    )


def _page(title: str, description: str, body: str, asset_prefix: str = "") -> str:
    """asset_prefix is the relative path back to OUTPUT_ROOT (e.g. "../../../"
    for a run page, "" for the top-level index), so every page links the one
    shared stylesheet/script instead of inlining its own copy -- the browser
    then caches both once instead of re-downloading them on every run/example
    page navigated to while browsing the archive."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="{esc(description)}">
<title>{esc(title)}</title>
{FONT_LINK}
<link rel="stylesheet" href="{asset_prefix}assets/style.css"></head><body>
{body}
<script src="{asset_prefix}assets/viewer.js" defer></script>
</body></html>"""


# ---------------------------------------------------------------------------
# Per-run pages
# ---------------------------------------------------------------------------


def _run_links(commit: str, run_id: str) -> tuple[str, str]:
    commit_url = (
        os.environ.get("REPORT_COMMIT_URL")
        or f"https://github.com/raiyiz/catsy/commit/{commit}"
    )
    ci_url = os.environ.get("REPORT_CI_URL") or "https://github.com/raiyiz/catsy/actions"
    return commit_url, ci_url


def build_rich_run_page(
    site_run_root: Path,
    spec: ExampleSpec,
    config: dict[str, object],
    commit: str,
    ref: str,
    run_id: str,
    generated_at: str,
) -> None:
    short_commit = commit[:12]
    plots = (
        sorted((site_run_root / "plots").glob("*.png"))
        if (site_run_root / "plots").exists()
        else []
    )
    by_stem = {plot.stem: plot for plot in plots}
    journals = sorted(
        path
        for path in site_run_root.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"}
    )

    pipeline, stages = [], []
    for number, title, category, description, plot_stem, inspect, result in STAGES:
        pipeline.append(
            f'<a class="pipeline-step" href="#stage-{esc(number)}" style="--step-color:var(--{esc(category)})">'
            f'<span class="n">{esc(number)}</span><strong>{esc(title)}</strong>'
            f"<small>{esc(CATEGORY_LABELS[category])}</small></a>"
        )
        plot = by_stem.get(plot_stem) if plot_stem else None
        if plot:
            relative = esc(plot.relative_to(site_run_root).as_posix())
            visual = (
                f'<button class="stage-plot" data-src="{relative}" data-title="{esc(title)}" type="button">'
                f'<img src="{relative}" alt="{esc(title)}" loading="lazy"></button>'
            )
        else:
            visual = '<div class="stage-plot empty"><span>—</span><small>response recorded in journal, not plotted</small></div>'
        insight = INSIGHTS.get(number, "")
        insight_html = (
            f'<div class="insight"><strong>Why it matters</strong>{esc(insight)}</div>'
            if insight
            else ""
        )
        stages.append(
            f'<article class="stage" data-category="{esc(category)}" id="stage-{esc(number)}">'
            f'<div class="stage-index" style="--stage-color:var(--{esc(category)})">'
            f'<span class="num">{esc(number)}</span><span class="cat">{esc(CATEGORY_LABELS[category])}</span></div>'
            f"{visual}"
            f'<div class="stage-copy" style="--stage-color:var(--{esc(category)})">'
            f"<h3>{esc(title)}</h3><p>{esc(description)}</p>"
            f'<div class="callout"><span class="lbl">Look for</span><span>{esc(inspect)}</span></div>'
            f'<div class="callout"><span class="lbl">Diagnostic</span><span>{esc(result)}</span></div>'
            f"{insight_html}</div></article>"
        )

    journal_links = _journal_links(journals, site_run_root)
    filters = "".join(
        f'<button class="filter" data-filter="{esc(key)}" type="button">{esc(label)}</button>'
        for key, label in CATEGORY_LABELS.items()
    )
    commit_url, ci_url = _run_links(commit, run_id)
    circuit_svg = render_circuit_diagram(circuit_ops_for(config), with_readout=True)

    body = f"""
{
        _topbar(
            [("catsy · lab", "../../../"), (spec.title, "../../"), (short_commit, "")],
            [
                ("all examples", "../../../"),
                ("this example", "../../"),
                ("pipeline", "#pipeline"),
                ("stages", "#stages"),
                ("journal", "#journal"),
            ],
        )
    }
<main>
<section class="hero"><div class="container">
  <div class="eyebrow">{esc(spec.title)} · {esc(generated_at)}</div>
  <h1>Gaussian preparation, non-Gaussian processing, interferometry and readout.</h1>
  <p class="lead">{
        esc(spec.tagline)
    } Each diagnostic sits beside the physical stage it documents.</p>
  <div class="meta">
    <span class="badge">commit <strong>{esc(short_commit)}</strong></span>
    <span class="badge">ref <strong>{esc(ref)}</strong></span>
    <span class="badge"><strong>{len(plots)}</strong> diagnostics</span>
    <span class="badge"><strong>{len(journals)}</strong> journal files</span>
  </div>
  <div class="links">
    <a class="button" href="{
        esc(commit_url)
    }" target="_blank" rel="noopener">source commit ↗</a>
    <a class="button" href="{esc(ci_url)}" target="_blank" rel="noopener">CI run ↗</a>
    <a class="button" href="../../">this example's runs</a>
    <a class="button" href="../../../">all examples</a>
  </div>
</div></section>

<section class="section alt" id="circuit"><div class="container">
  <div class="section-head"><div class="eyebrow">Circuit topology</div>
  <h2>What the three modes actually go through</h2>
  <p>Sourced from <span class="mono">build_circuit()</span> and this run's own config values —
  each wire is a mode starting in vacuum; each box is the operation applied to it, in execution order.</p></div>
  <div class="circuit-card">{circuit_svg}</div>
  <div class="circuit-legend">
    <span><span class="sw" style="background:var(--gaussian)"></span>Gaussian operation</span>
    <span><span class="sw" style="background:var(--measurement)"></span>measurement</span>
  </div>
</div></section>

<section class="section" id="pipeline"><div class="container">
  <div class="section-head"><div class="eyebrow">Experiment map</div>
  <h2>From state preparation to measurement</h2>
  <p>Each step links to the diagnostic and its physical interpretation below.</p></div>
  <div class="pipeline">{"".join(pipeline)}</div>
</div></section>

<section class="section alt" id="stages"><div class="container">
  <div class="section-head"><div class="eyebrow">Stage diagnostics</div>
  <h2>What happened at each step</h2>
  <p>Plots are shown once, beside the operation they document.</p></div>
  <div class="filters"><button class="filter active" data-filter="all" type="button">all</button>{
        filters
    }</div>
  <div class="stages">{"".join(stages)}</div>
</div></section>

<section class="section" id="journal"><div class="container">
  <div class="section-head"><div class="eyebrow">Reproducibility</div>
  <h2>Experiment journal</h2>
  <p>Machine-readable records remain beside the visual diagnostics.</p></div>
  <div class="journal">{journal_links}</div>
</div></section>
</main>

<div class="modal" id="viewer" role="dialog" aria-modal="true" aria-label="Plot viewer">
  <div class="modal-inner">
    <div class="modal-head"><span class="modal-title" id="viewer-title"></span>
    <button class="modal-close" id="viewer-close" type="button">close</button></div>
    <img id="viewer-image" alt="">
    <div class="modal-foot"><span>Esc to close · click outside to close</span>
    <a id="viewer-open" href="#" target="_blank" rel="noopener">open original ↗</a></div>
  </div>
</div>

<footer class="footer"><div class="container">
  {esc(spec.title)} · commit <span class="mono">{esc(commit)}</span><br>
  Static report generated by CI. Visualizations are produced through catsy's plotting helpers.
</div></footer>"""

    (site_run_root / "index.html").write_text(
        _page(
            f"{spec.title} · {short_commit}",
            f"Catsy {spec.title} report for {short_commit}",
            body,
            asset_prefix="../../../",
        ),
        encoding="utf-8",
    )


def build_journal_only_run_page(
    site_run_root: Path,
    spec: ExampleSpec,
    config: dict[str, object],
    commit: str,
    ref: str,
    run_id: str,
    generated_at: str,
) -> None:
    short_commit = commit[:12]
    journals = sorted(
        path
        for path in site_run_root.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"}
    )
    journal_links = _journal_links(journals, site_run_root)
    commit_url, ci_url = _run_links(commit, run_id)
    circuit_svg = render_circuit_diagram(circuit_ops_for(config), with_readout=False)

    metrics = _scalar_results_from_journal(journals)
    metrics_html = (
        "".join(
            f'<div class="metric"><div class="value">{esc(_format_metric(value))}</div>'
            f'<div class="label">{esc(key.replace("_", " "))}</div></div>'
            for key, value in metrics.items()
        )
        or '<div class="empty-state">No scalar results were logged for this run.</div>'
    )

    body = f"""
{
        _topbar(
            [("catsy · lab", "../../../"), (spec.title, "../../"), (short_commit, "")],
            [
                ("all examples", "../../../"),
                ("this example", "../../"),
                ("circuit", "#circuit"),
                ("results", "#results"),
                ("journal", "#journal"),
            ],
        )
    }
<main>
<section class="hero"><div class="container">
  <div class="eyebrow">{esc(spec.title)} · {esc(generated_at)}</div>
  <h1>The minimal three-mode Gaussian circuit.</h1>
  <p class="lead">{esc(spec.tagline)}</p>
  <div class="meta">
    <span class="badge">commit <strong>{esc(short_commit)}</strong></span>
    <span class="badge">ref <strong>{esc(ref)}</strong></span>
    <span class="badge"><strong>{len(journals)}</strong> journal files</span>
  </div>
  <div class="links">
    <a class="button" href="{
        esc(commit_url)
    }" target="_blank" rel="noopener">source commit ↗</a>
    <a class="button" href="{esc(ci_url)}" target="_blank" rel="noopener">CI run ↗</a>
    <a class="button" href="../../">this example's runs</a>
    <a class="button" href="../../../">all examples</a>
  </div>
</div></section>

<section class="section alt" id="circuit"><div class="container">
  <div class="section-head"><div class="eyebrow">Circuit topology</div>
  <h2>Build, run, persist</h2>
  <p>The same three-mode circuit as the complex simulation example, without the Fock-space
  processing or measurement stages that follow it there — it stops after the Gaussian circuit
  and logs the result to the journal.</p></div>
  <div class="circuit-card">{circuit_svg}</div>
  <div class="circuit-legend">
    <span><span class="sw" style="background:var(--gaussian)"></span>Gaussian operation</span>
  </div>
</div></section>

<section class="section" id="results"><div class="container">
  <div class="section-head"><div class="eyebrow">Logged metrics</div>
  <h2>Scalar results from this run</h2></div>
  <div class="metrics-strip">{metrics_html}</div>
</div></section>

<section class="section alt" id="journal"><div class="container">
  <div class="section-head"><div class="eyebrow">Reproducibility</div>
  <h2>Experiment journal</h2>
  <p>The circuit, final state, and metrics above are all recorded here.</p></div>
  <div class="journal">{journal_links}</div>
</div></section>
</main>

<footer class="footer"><div class="container">
  {esc(spec.title)} · commit <span class="mono">{esc(commit)}</span><br>
  Static report generated by CI.
</div></footer>"""

    (site_run_root / "index.html").write_text(
        _page(
            f"{spec.title} · {short_commit}",
            f"Catsy {spec.title} report for {short_commit}",
            body,
            asset_prefix="../../../",
        ),
        encoding="utf-8",
    )


def _journal_links(journals: list[Path], site_run_root: Path) -> str:
    if not journals:
        return '<div class="empty-state">No journal files were generated.</div>'
    links = []
    for journal in journals:
        relative = esc(journal.relative_to(site_run_root).as_posix())
        links.append(
            f'<a class="journal-file" href="{relative}" target="_blank" rel="noopener">'
            f'<span class="file-type">{esc(journal.suffix[1:].upper())}</span>'
            f"<span><strong>{relative}</strong><small>{journal.stat().st_size:,} bytes · open raw file ↗</small></span></a>"
        )
    return "".join(links)


def _scalar_results_from_journal(journals: list[Path]) -> dict[str, object]:
    for journal in journals:
        if journal.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(journal.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for run in data.get("runs", []):
            results = run.get("scalar_results")
            if results:
                return dict(results)
    return {}


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# ---------------------------------------------------------------------------
# Per-example archive page
# ---------------------------------------------------------------------------


def _read_metadata(run_dir: Path) -> dict[str, str]:
    metadata_file = run_dir / "run_metadata.txt"
    values: dict[str, str] = {}
    if metadata_file.exists():
        for line in metadata_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                values[key] = value
    return values


def _run_datetime(run_dir: Path) -> str:
    timestamp = _read_metadata(run_dir).get("timestamp")
    if timestamp:
        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime(
                "%Y-%m-%d · %H:%M UTC"
            )
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC).strftime(
            "%Y-%m-%d · %H:%M UTC"
        )
    except OSError:
        return "time unavailable"


def _example_meta(site_example_root: Path) -> dict[str, str]:
    meta_file = site_example_root / "example_meta.json"
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    humanized = site_example_root.name.replace("-", " ").title()
    return {"title": humanized, "tagline": "", "kind": "rich"}


def build_example_archive_page(site_example_root: Path) -> None:
    meta = _example_meta(site_example_root)
    runs_root = site_example_root / "runs"
    run_dirs = (
        sorted(
            (p for p in runs_root.iterdir() if p.is_dir()),
            key=lambda p: _read_metadata(p).get("timestamp", p.name),
            reverse=True,
        )
        if runs_root.exists()
        else []
    )
    cards = "".join(
        f'<a class="run-card" href="runs/{esc(p.name)}/"><span class="run-dot"></span>'
        f'<span class="run-copy"><small class="run-time">{esc(_run_datetime(p))}</small>'
        f"<strong>{esc(p.name[:12])}</strong>"
        f'<span class="run-detail">open run explorer →</span></span></a>'
        for p in run_dirs
    )
    body = f"""
{_topbar([("catsy · lab", "../"), (meta["title"], "")], [("all examples", "../")])}
<main><div class="container" style="padding:64px 0 80px">
  <div class="eyebrow">catsy · commit-addressed run history</div>
  <h1 style="font-size:clamp(32px,5vw,48px);margin-top:10px">{esc(meta["title"])}</h1>
  <p class="lead" style="margin-top:14px">{esc(meta.get("tagline", ""))}</p>
  <div class="archive-list">{cards or '<div class="empty-state">No runs yet.</div>'}</div>
</div></main>"""
    (site_example_root / "index.html").write_text(
        _page(
            f"{meta['title']} · runs",
            f"Run history for {meta['title']}",
            body,
            asset_prefix="../",
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Top-level index
# ---------------------------------------------------------------------------


def build_top_level_index(output_root: Path) -> None:
    example_dirs = sorted(
        p for p in output_root.iterdir() if p.is_dir() and (p / "runs").exists()
    )
    cards = []
    for example_dir in example_dirs:
        meta = _example_meta(example_dir)
        runs_root = example_dir / "runs"
        run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
        latest = max((_run_datetime(p) for p in run_dirs), default="no runs yet")
        cards.append(
            f'<a class="example-card" href="{esc(example_dir.name)}/">'
            f'<div class="kind">{esc(meta.get("kind", "rich")).replace("_", " ")}</div>'
            f"<h2>{esc(meta['title'])}</h2><p>{esc(meta.get('tagline', ''))}</p>"
            f'<div class="stat-row"><span><strong>{len(run_dirs)}</strong> run(s)</span>'
            f"<span>latest: <strong>{esc(latest)}</strong></span></div></a>"
        )

    body = f"""
{_topbar([("catsy · lab", "")], [])}
<main><div class="container" style="padding:64px 0 80px">
  <div class="eyebrow">catsy · commit-addressed simulation archive</div>
  <h1 style="font-size:clamp(36px,6vw,58px);margin-top:10px">Example runs</h1>
  <p class="lead" style="margin-top:14px">Each example script gets its own section and keeps its own
  run history. Open a section to browse its runs; open a run to follow the state transformation
  stage by stage.</p>
  <div class="example-grid">{"".join(cards) or '<div class="empty-state">No example output has been published yet.</div>'}</div>
</div></main>"""
    (output_root / "index.html").write_text(
        _page("catsy · lab", "Catsy example run archive", body), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _write_shared_assets(output_root: Path) -> None:
    """Write the one shared stylesheet/script every page links to. Written
    once per build rather than inlined per page, so a browser walking the
    archive (many run + example pages) fetches each only once and caches it,
    instead of re-downloading an identical copy on every navigation."""
    assets_dir = output_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "style.css").write_text(PAGE_STYLE, encoding="utf-8")
    (assets_dir / "viewer.js").write_text(RUN_SCRIPT, encoding="utf-8")


def main() -> None:
    commit = os.environ.get("REPORT_COMMIT", os.environ.get("GITHUB_SHA", "unknown"))
    ref = os.environ.get("REPORT_REF", "unknown")
    run_id = os.environ.get("REPORT_RUN_ID", "")
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d · %H:%M UTC")
    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    _write_shared_assets(OUTPUT_ROOT)

    built_any = False
    for spec in EXAMPLES:
        if not spec.run_root.exists():
            continue
        built_any = True

        site_example_root = OUTPUT_ROOT / spec.id
        site_run_root = site_example_root / "runs" / commit
        if site_run_root.exists():
            shutil.rmtree(site_run_root)
        site_run_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(spec.run_root, site_run_root)

        (site_run_root / "run_metadata.txt").write_text(
            f"timestamp={timestamp}\ncommit={commit}\nref={ref}\nrun_id={run_id}\n",
            encoding="utf-8",
        )

        config = _read_run_config(spec.config_path)
        if spec.kind == "rich":
            build_rich_run_page(
                site_run_root, spec, config, commit, ref, run_id, generated_at
            )
        else:
            build_journal_only_run_page(
                site_run_root, spec, config, commit, ref, run_id, generated_at
            )

        site_example_root.mkdir(parents=True, exist_ok=True)
        (site_example_root / "example_meta.json").write_text(
            json.dumps({"title": spec.title, "tagline": spec.tagline, "kind": spec.kind}),
            encoding="utf-8",
        )

    if not built_any:
        raise SystemExit(
            "No example output directories found (expected one of: "
            + ", ".join(str(spec.run_root) for spec in EXAMPLES)
            + ")."
        )

    if OUTPUT_ROOT.exists():
        for entry in sorted(OUTPUT_ROOT.iterdir()):
            if entry.is_dir() and (entry / "runs").exists():
                build_example_archive_page(entry)

    build_top_level_index(OUTPUT_ROOT)


if __name__ == "__main__":
    main()

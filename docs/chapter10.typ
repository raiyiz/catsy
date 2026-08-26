#import "@preview/physica:0.9.8": *
#import "links.typ": src-link

// ==========================================
// CHAPTER 10
// ==========================================
= Chapter 10: Practical Guide

This closing chapter summarizes the previous nine chapters into a practical overview: how to import the package, how the modules relate to each other, which conventions apply package-wide, and how to run the test suite. The emphasis on explicit conventions, tests, and reproducible workflows is consistent with established guidance for scientific computing, including #link("https://doi.org/10.1371/journal.pcbi.1005510")[Wilson et al. (2017)].

== Module overview

`catsy` is deliberately organized into a small number of clearly scoped modules. QuTiP is a hard runtime dependency, not an optional extra — it is already loaded when `catsy` is imported.

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: 0.5pt + gray.lighten(40%),
  [*Module*], [*Contents*],
  [#src-link("src/catsy/core.py")], [Symplectic form $Omega$, validation helpers, Williamson decomposition, JSON helper functions (Chapters 1, 5).],
  [#src-link("src/catsy/gaussian/__init__.py")], [`GaussianState`, `GaussianChannel`/`LossChannels`, `GaussianMeasurements`, phase-space analysis (Chapters 1–6).],
  [#src-link("src/catsy/gaussian/visualization.py")], [Gaussian-state plots, composite dashboards (`plot_state_dashboard`, `plot_evolution`, `plot_multimode_evolution`), and animations (Chapter 6).],
  [#src-link("src/catsy/fock.py")], [`FockGates`: photon addition/subtraction on QuTiP states (Chapter 7).],
  [#src-link("src/catsy/fock_visualization.py")], [Photon-number statistics, Fock-coherence, and Wigner plots for QuTiP states (Chapter 6).],
  [#src-link("src/catsy/visualization.py")], [Plotting primitives (figure lifecycle, phase-space styling) shared by the two visualization modules above.],
  [#src-link("src/catsy/optics.py")], [`Circuit`/`Mode` (generic executable gate sequence, Chapter 3), `KerrCavity`/`MachZehnderInterferometer`: time-resolved QuTiP simulations (Chapter 7). Reusable Gaussian gate layouts live on `Circuit` itself (Chapter 8).],
  [#src-link("src/catsy/journal.py")], [`JournalEntry`/`SimulationJournal`: experiment persistence (Chapter 9).],
)

There is no separate compatibility-shim or simulation-only module: `FockGates` lives in `catsy.fock`, and `KerrCavity`/`MachZehnderInterferometer` live in `catsy.optics`, since both model specific pieces of optical hardware rather than generic phase-space transformations. Imports happen either from the individual modules or from the public names re-exported by `catsy/__init__.py`:

```python
from catsy import (
    GaussianState, GaussianChannel, LossChannels,
    Circuit, Mode, GaussianMeasurements,
    compute_wigner_analytically, compute_joint_correlation, compute_duan_inseparability,
    FockGates, KerrCavity, MachZehnderInterferometer,
    JournalEntry, SimulationJournal,
)
```

== Package-wide conventions

All modules share the same underlying physical conventions, regardless of which layer is being used:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: 0.5pt + gray.lighten(40%),
  [*Convention*], [*Value / definition*],
  [Quadrature ordering], [$(q_1, p_1, q_2, p_2, dots)$ -- interleaved, not block-wise.],
  [Units], [$hbar = 1$, $[q, p] = i$.],
  [Vacuum covariance], [$V_"vac" = 1/2 bb(1)$.],
  [Displacement / amplitude], [$alpha = (x + i p) / sqrt(2)$.],
  [Covariance definition], [symmetrized second moments: $V_(i j) = 1/2 chevron.l \{r_i - d_i, r_j - d_j\} chevron.r$.],
  [Beam splitter], [power transmissivity $eta$, see Chapter 2.],
)

For a single-mode squeezed vacuum state with squeezing strength $r$ and $theta = 0$, this correspondingly gives $"Var"(q) = e^(-2r)/2$ and $"Var"(p) = e^(2r)/2$ — the reference values against which `GaussianState.squeeze` (Chapter 2) and the Wigner diagnostics (Chapter 6) are verified in the test suite.

== Two typical workflows

*Declarative, via `Circuit` (Chapter 3):* a `Circuit` describes the ordered gate sequence, and `run` executes it against an explicitly supplied initial state.

```python
from catsy import Circuit, Gate, GaussianState, loss

initial = GaussianState.tmsv("a", "b", r=0.7)
circuit = Circuit()
a = circuit.mode("a")
circuit.mode("b")
circuit.add_gate(Gate(name="Noise", transform=loss, modes=(a.name,), kwargs={"eta": 0.9}))
final = circuit.run(initial)
```

*Direct, gate by gate (Chapters 2 and 5):* for exploratory use, where every intermediate state should be inspected.

```python
from catsy import GaussianState

state = GaussianState.vacuum(("a",))
state = state.squeeze("a", r=0.5)
state = state.displace("a", alpha=0.4 + 0.2j)
```

Both paths produce identical `GaussianState` objects and can be freely mixed: a directly constructed state can be fed as `initial_state` into `run` (as in the first example), and a compiled final state can subsequently be processed further with `GaussianState` methods directly.

== Test suite

The test suite emphasizes both physical invariants (uncertainty relation, symplectic conservation, exact loss limiting cases) and analytical reference values across the entire module chain -- from state validation through to the Gaussian-Fock boundary:

```bash
uv run pytest
```

Plot-generating tests (`plot_covariance_matrix`, `plot_wigner`, `plot_joint_correlation`) are deliberately opt-in, to keep the default suite headless-friendly and fast:

```bash
uv run pytest --plot
```

== Scope and boundaries

`catsy` is deliberately a focused tool, not a full quantum-computing framework. Priority is given to readable CV quantum-optics mathematics, explicit conventions, and small, composable building blocks -- over the broadest possible gate catalog. Where Fock-space physics is required (Chapter 7), the package deliberately delegates to QuTiP rather than maintaining its own, redundant Hilbert-space layer.

---


=== Literature and reproducibility
The practical workflow recommended here is supported by:

- #link("https://doi.org/10.1371/journal.pcbi.1005510")[G. Wilson et al., “Good enough practices in scientific computing,” *PLoS Computational Biology* 13, e1005510 (2017).]
- #link("https://doi.org/10.1038/sdata.2016.18")[M. D. Wilkinson et al., “The FAIR Guiding Principles for scientific data management and stewardship,” *Scientific Data* 3, 160018 (2016).]

For the underlying continuous-variable physics, the recommended entry points are #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and #link("https://www.routledge.com/Quantum-Continuous-Variables-A-Primer-of-Theoretical-Methods/Serafini/p/book/9781032157238")[Serafini (2023)].

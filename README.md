
# catsy

`catsy` is a Python toolkit for continuous-variable quantum optics, built around Gaussian states in phase space.

The main idea is to keep Gaussian calculations in phase space where possible, describe experiments as reusable circuits, and move to a truncated Fock-space representation when needed.


### For a more detailed documentation, check the latest build of the [Specifications](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst)


### Status

[![Pipeline status](https://gitlab.uni-hannover.de/inl/catsy/badges/main/pipeline.svg)](https://gitlab.uni-hannover.de/inl/catsy/-/pipelines) [![Coverage](https://gitlab.uni-hannover.de/inl/catsy/badges/main/coverage.svg)](https://inl.idmpages.uni-h.de/catsy/)

***⚠ This package was build with heavy use of AI***

---

The default branch publishes the latest interactive HTML coverage report through GitLab Pages. Every pipeline (including merge requests) also attaches its own HTML coverage report directly to the `pytest` job -- open the job and use the "HTML coverage report" artifact link if the default-branch Pages link above is ever out of date. The CI pipeline also uploads the Cobertura XML report so GitLab can show the coverage percentage and line-by-line coverage annotations in merge requests.


## Quick start

```python
from catsy import Circuit, GaussianState, Gate, loss
from catsy.gaussian.visualization import plot_covariance_matrix

initial = GaussianState.tmsv("a", "b", r=0.7)

noise = Gate(
    name="Noise",
    transform=loss,
    modes=("a",),
    kwargs={"eta": 0.9},
)

circuit = Circuit().add_mode("a").add_mode("b")
circuit.add_gate(noise)

final = circuit.run(initial)
plot_covariance_matrix(final)
```

Here `r` is the squeezing strength and `eta` is the power transmissivity of the loss channel.

`add_mode("a")` above is a shorthand for `circuit.mode("a")`, which registers the mode and returns a `Mode` handle owned by that circuit:

```python
circuit = Circuit()
a = circuit.mode("a")
b = circuit.mode("b")
circuit.squeeze(a, r=0.5).beam_splitter(a, b, eta=0.5)
```

Building gates from these handles instead of bare strings means a mode meant for one circuit can't accidentally be wired into a different one -- passing a `Mode` owned by another circuit (or a free `Mode(name)` with no owner) raises immediately. Plain mode-name strings still work anywhere a registered mode is expected, as in the quick start above.
## Where to start
| If you want to...                           | Use                               |
| ------------------------------------------- | --------------------------------- |
| Create Gaussian states                      | [`GaussianState`](src/catsy/gaussian/__init__.py#L83) |
| Apply Gaussian operations                   | [`GaussianState`](src/catsy/gaussian/__init__.py#L83) |
| Build a circuit / define an optical layout  | [`Circuit`](src/catsy/optics.py#L100), [`Mode`](src/catsy/optics.py#L68) |
| Model loss and thermal noise                | [`LossChannels`](src/catsy/gaussian/__init__.py#L497), [`GaussianChannel`](src/catsy/gaussian/__init__.py#L434) |
| Perform homodyne or heterodyne measurements | [`GaussianMeasurements`](src/catsy/gaussian/__init__.py#L623) |
| Inspect a covariance matrix                 | [`GaussianState`](src/catsy/gaussian/__init__.py#L83) |
| Calculate a Wigner function                 | [`compute_wigner_analytically()`](src/catsy/gaussian/__init__.py#L750) |
| Convert to Fock space                       | [`GaussianState.to_qutip()`](src/catsy/gaussian/__init__.py#L289) |
| Visualize a state or its evolution          | [`plot_state_dashboard()`](src/catsy/gaussian/visualization.py#L671), [`plot_evolution()`](src/catsy/gaussian/visualization.py#L595) |
| Visualize a truncated Fock-space state      | [`plot_fock_dashboard()`](src/catsy/fock/visualization.py) |
| Save states and experiments                 | [`SimulationJournal`](src/catsy/journal.py#L355) |
## Gaussian states

States are represented in phase space by their first moments and covariance matrix. Common operations include:

* vacuum, coherent, and two-mode squeezed vacuum states
* squeezing and displacement
* phase shifts and beam splitters
* loss and thermal channels
* homodyne and heterodyne measurements

For example:

```python
from catsy import GaussianState
state = GaussianState.vacuum(("a",))
state = state.squeeze("a", r=0.5)
state = state.displace(
    "a",
    alpha=0.4 + 0.2j,
)
```
## Circuits

For a sequence of operations that you want to keep as a reusable experiment, `Circuit` provides an executable sequence independent of the Gaussian state implementation:

```python
from catsy import Circuit, GaussianState

initial = GaussianState.vacuum(("a", "b"))
circuit = Circuit().add_mode("a").add_mode("b")
circuit.squeeze("a", r=0.7, theta=0.0)
circuit.beam_splitter("a", "b", eta=0.5)
final = circuit.run(initial)
```

Circuits can also be serialized and restored, and rendered as a plain-text schematic:

```python
circuit.render_schematic()   # -> str
circuit.draw()                # prints it
```

## Fock-space calculations

Gaussian states can be converted to QuTiP density matrices when an explicit Fock-space representation is useful:

```python
rho = state.to_qutip(N_cutoff=20)
```

The cutoff is numerical, so it should be increased until the quantity of interest has converged.

`catsy` uses [QuTiP](https://qutip.org/) for this part of the calculation.

## Conventions

The phase-space convention used throughout the package is

* $\hbar = 1$
* $[x,p] = i$
* $V_\mathrm{vac} = I/2$
* quadratures ordered as `(x1, p1, x2, p2, ...)`
* $\alpha = (x + ip)/\sqrt{2}$

For a single-mode squeezed vacuum with $\theta=0$,

$$
\operatorname{Var}(x) = \frac{e^{-2r}}{2},
\qquad
\operatorname{Var}(p) = \frac{e^{2r}}{2}.
$$

These conventions are used consistently by the Gaussian and Fock-space interfaces.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://gitlab.uni-hannover.de/inl/catsy.git
cd catsy
uv sync
```

Run Python or the test suite through the project environment:

```bash
uv run python
uv run pytest
```

For tests involving plots:

```bash
uv run pytest --plot
```

## Project structure

| Module                                                                                                       | Contents                                                            |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [`core.py`](src/catsy/core.py)         | conventions, validation, numerical helpers                          |
| [`gaussian/`](src/catsy/gaussian/__init__.py) | states, operations, channels, measurements                   |
| [`gaussian/visualization.py`](src/catsy/gaussian/visualization.py) | Gaussian-state plots, dashboards, and animations |
| [`fock.py`](src/catsy/fock.py)         | Fock-space functionality                                             |
| [`fock/visualization.py`](src/catsy/fock/visualization.py) | photon-statistics, Fock-coherence, and Wigner plots for QuTiP states |
| [`visualization.py`](src/catsy/visualization.py) | shared plotting primitives (figure lifecycle, phase-space styling, annotation, and colorbar helpers) used by both visualization modules above |
| [`optics.py`](src/catsy/optics.py)     | circuits, modes, QuTiP-based cavity/interferometer simulations       |
| [`journal.py`](src/catsy/journal.py)   | experiment persistence                                               |
## Documentation

The [documentation](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst) contains the more detailed API and usage information.

The repository also contains examples and tests that can be useful when exploring particular operations.

## Status

`catsy` is focused on continuous-variable quantum optics rather than providing a general-purpose quantum-computing framework. The API is still developing, but the core conventions and numerical operations are covered by the test suite.


<sub>cats & states & oha & phos & nothingness, very pur so</sub>

## Test coverage

Coverage is collected with [`pytest-cov`](https://pytest-cov.readthedocs.io/) and is intentionally configured as a development concern rather than a runtime dependency. Branch coverage (not just line coverage) is enabled via `[tool.coverage.run]` in `pyproject.toml`. Locally, run:

If the development dependencies have changed, refresh the lockfile first with `uv lock`. Then run:

```bash
uv sync --group dev
uv run pytest --cov=src/catsy --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml
```

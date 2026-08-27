<sub><sub>cats & states & oha & phos & nothingness, very purr so</sub></sub>
\
\
# catsy

`catsy` is a Python toolkit for continuous-variable quantum optics. It provides a compact set of tools for building, manipulating, simulating, visualizing, and persisting quantum optical states and experiments.

The central design choice is to keep calculations in the representation that is most natural for them:

* **Gaussian states** are represented directly in phase space by their first moments and covariance matrices.
* **Circuits** describe a sequence of optical transformations independently of any particular input state.
* **Fock-space calculations** use QuTiP when a truncated Hilbert-space representation is needed, for example for non-Gaussian states or photon-number observables.
* **Visualization and journaling** provide common ways to inspect and persist the results of either representation.

This means a typical workflow can remain entirely Gaussian until a Fock-space representation is actually useful:

```python
from catsy import Circuit, GaussianState

initial = GaussianState.tmsv("a", "b", r=0.7)

circuit = (
    Circuit()
    .add_mode("a")
    .add_mode("b")
)
circuit.squeeze("a", r=0.5)
circuit.beam_splitter("a", "b", eta=0.5)

final = circuit.run(initial)
```

## States, modes, and circuits

There are two related ideas to keep separate:

**A mode name** is simply a string such as `"a"` or `"signal"`. Mode names are what appear in serialized data and in the state representation.

**A `Mode` object** is a runtime handle belonging to a particular `Circuit`. It lets the circuit API check that a mode is being used with the circuit that owns it.

You normally do not need to construct `Mode` objects yourself. A circuit can create one for you:

```python
circuit = Circuit()

a = circuit.mode("a")
b = circuit.mode("b")

circuit.squeeze(a, r=0.5)
circuit.beam_splitter(a, b, eta=0.5)
```

The `mode()` method registers the mode **and returns its `Mode` handle**.

For fluent construction, `add_mode()` is a convenience method that performs the same registration but returns the **circuit itself**, so calls can be chained:

```python
circuit = (
    Circuit()
    .add_mode("a")
    .add_mode("b")
    .add_mode("reference")
)
```

The two methods therefore have different purposes:

| Method                  | Registers the mode | Returns                    |
| ----------------------- | ------------------ | -------------------------- |
| `circuit.mode("a")`     | Yes                | the new `Mode` handle      |
| `circuit.add_mode("a")` | Yes                | the `Circuit` for chaining |

Gate methods such as `squeeze()`, `rotate()`, `displace()`, and `beam_splitter()` accept either a registered mode name or the corresponding `Mode` handle. Using handles enables ownership checking:

```python
first = Circuit().add_mode("a")
second = Circuit().add_mode("a")

a = first.mode("a")
second.squeeze(a, r=0.5)  # rejected: `a` belongs to another circuit
```

Plain strings remain convenient when the circuit is small and the mode ownership is obvious:

```python
circuit.squeeze("a", r=0.5)
circuit.beam_splitter("a", "b", eta=0.5)
```

The string identifies a mode by **name**, while `Mode` identifies the particular runtime mode owned by a circuit.

## Where to start
| If you want to...                           | Use                               |
| ------------------------------------------- | --------------------------------- |
| Create Gaussian states                      | [`GaussianState`](src/catsy/gaussian/__init__.py#L83) |
| Apply Gaussian operations                   | [`GaussianState`](src/catsy/gaussian/__init__.py#L83) |
| Build a circuit / define an optical layout  | [`Circuit`](src/catsy/optics.py#L123), [`Mode`](src/catsy/optics.py#L66) |
| Model loss and thermal noise                | [`LossChannels`](src/catsy/gaussian/__init__.py#L558), [`GaussianChannel`](src/catsy/gaussian/__init__.py#L495) |
| Perform homodyne or heterodyne measurements | [`GaussianMeasurements`](src/catsy/gaussian/__init__.py#L689) |
| Inspect a covariance matrix                 | [`GaussianState`](src/catsy/gaussian/__init__.py#L83) |
| Calculate a Wigner function                 | [`compute_wigner_analytically()`](src/catsy/gaussian/__init__.py#L816) |
| Convert to Fock space                       | [`GaussianState.to_qutip()`](src/catsy/gaussian/__init__.py#L350) |
| Visualize a state or its evolution          | [`plot_state_dashboard()`](src/catsy/gaussian/visualization.py#L670), [`plot_evolution()`](src/catsy/gaussian/visualization.py#L594) |
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
| [`fock.py`](src/catsy/fock/__init__.py)         | Fock-space functionality                                             |
| [`fock/visualization.py`](src/catsy/fock/visualization.py) | photon-statistics, Fock-coherence, and Wigner plots for QuTiP states |
| [`visualization.py`](src/catsy/visualization.py) | shared plotting primitives (figure lifecycle, phase-space styling, annotation, and colorbar helpers) used by both visualization modules above |
| [`optics.py`](src/catsy/optics.py)     | circuits, modes, QuTiP-based cavity/interferometer simulations       |
| [`journal.py`](src/catsy/journal.py)   | experiment persistence                                               |
## Documentation

The [documentation](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst) contains the more detailed API and usage information.

The repository also contains examples and tests that can be useful when exploring particular operations.

## Status

`catsy` is focused on continuous-variable quantum optics rather than providing a general-purpose quantum-computing framework. The API is still developing, but the core conventions and numerical operations are covered by the test suite.



## Test coverage

Coverage is collected with [`pytest-cov`](https://pytest-cov.readthedocs.io/) and is intentionally configured as a development concern rather than a runtime dependency. Branch coverage (not just line coverage) is enabled via `[tool.coverage.run]` in `pyproject.toml`. Locally, run:

If the development dependencies have changed, refresh the lockfile first with `uv lock`. Then run:

```bash
uv sync --group dev
uv run pytest --cov=src/catsy --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml
```

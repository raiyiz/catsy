<sub><sub>cats & states & oha & phos & nothingness, very purr so</sub></sub>




# catsy

`catsy` is a Python toolkit for continuous-variable quantum optics. Its primary abstraction is the **quantum state**: states are created, transformed, measured, visualized, and, when necessary, converted between different representations.

The main representation is the **Gaussian state**. A `GaussianState` describes a continuous-variable quantum state through its first moments and covariance matrix in phase space. This makes the common operations of Gaussian quantum optics—such as squeezing, displacement, phase shifts, beam splitters, loss, and thermal noise—compact and efficient to represent and manipulate.

For example:

```python
from catsy import GaussianState

state = GaussianState.vacuum(("signal",))

state = state.squeeze("signal", r=0.6)
state = state.displace("signal", alpha=0.4 + 0.2j)
state = state.rotate("signal", phi=0.3)
```

A state can also start from a physically meaningful Gaussian construction:

```python
from catsy import GaussianState

state = GaussianState.coherent("signal", alpha=1.2 + 0.4j)
state = GaussianState.tmsv("signal", "idler", r=0.7)
```

The same `GaussianState` interface is used for both single- and multi-mode states. Internally, the state stores its mode ordering, displacement vector, and covariance matrix. 

This phase-space representation is the default because many optical calculations can be performed without introducing a truncated Hilbert space.

### Conventions

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

## From states to experiments

When several operations should be treated as one reusable experiment, `Circuit` provides an executable sequence of transformations. A circuit describes **what happens** to a state; the `GaussianState` describes **what state is actually being transformed**.

```python
from catsy import Circuit, GaussianState

initial = GaussianState.tmsv("signal", "idler", r=0.7)

circuit = Circuit()
circuit.add_mode("signal").add_mode("idler")

circuit.squeeze("signal", r=0.4)
circuit.beam_splitter("signal", "idler", eta=0.5)
circuit.loss("signal", eta=0.9)

final = circuit.run(initial)
```

This separation is intentional:

```text
GaussianState
     │
     │  input
     ▼
  Circuit
     │
     │  sequence of transformations
     ▼
GaussianState
```

States are represented in phase space by their first moments and covariance matrix. Common operations include:

* vacuum, coherent, and two-mode squeezed vacuum states
* squeezing and displacement
* phase shifts and beam splitters
* loss and thermal channels
* homodyne and heterodyne measurements

A circuit therefore does not replace a state, and a state does not have to be embedded in a circuit. You can manipulate a state directly for one-off calculations, or construct a circuit when the sequence of operations itself is something you want to reuse, inspect, serialize, or run with different initial states.

## When Gaussian states are not enough

Gaussian states are not a universal representation. Some experiments produce or require genuinely non-Gaussian physics, where a covariance matrix alone cannot capture the state.

For these cases, `catsy` can convert a Gaussian state into a **truncated Fock-space representation** backed by [QuTiP](https://qutip.org/):

```python
rho = final.to_qutip(N_cutoff=30)
```

The resulting density matrix can then be used with Fock-space operations and observables such as photon-number statistics, parity, or non-Gaussian state transformations.

The cutoff is numerical, so it should be increased until the quantity of interest has converged.

This gives the overall workflow:

```text
construct state
      │
      ▼
Gaussian phase-space operations
      │
      ▼
optional circuit composition
      │
      ▼
Gaussian result
      │
      ├──────────────► visualize / measure / journal
      │
      ▼
 optional Fock conversion
      │
      ▼
truncated Hilbert-space calculations
```

Most users can therefore stay entirely within the Gaussian representation until they actually need Fock-space physics.

## Modes and circuits

Circuits can be serialized and restored, and rendered as a plain-text schematic:

```python
circuit.render_schematic()      # -> str
circuit.draw()                  # prints it
```

A circuit operates on named optical modes such as `"signal"` and `"idler"`. In the simplest code, these names are just strings:

```python
circuit.squeeze("signal", r=0.5)
circuit.beam_splitter("signal", "idler", eta=0.5)
```

In general though, modes belong to a circuit invoked with `Circuit.mode()` to prevent mixing up modes from different circuits. `add_mode()` is the fluent convenience form: it registers the mode and returns the circuit so that mode registration can be chained. `mode()` registers the mode and returns the `Mode` handle itself. Those distinguish beam paths, ports, channels.

```python
circuit = Circuit()
signal = circuit.mode("signal")
idler = circuit.mode("idler")

circuit.squeeze(signal, r=0.5)
circuit.beam_splitter(signal, idler, eta=0.5)
```


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

## Simulation explorer

### A visual laboratory notebook

The complex example is continuously executed in CI as a small, reproducible experiment. It deliberately crosses the package's main physical layers:

```text
┌─────────────────────┐
│ Gaussian preparation│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3-mode optical      │──────► covariance & correlations
│ circuit             │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Fock-space bridge   │
│ even cat state      │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ photon subtraction  │
│ photon addition     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ lossy Mach–Zehnder  │
│ phase scan          │
└──────────┬──────────┘
           ├──────────────► homodyne
           └──────────────► heterodyne
```
<details>
<summary>
Every published run is tied to its exact source commit and keeps its **plots + machine-readable journal together**.
</summary>

- stage-by-stage navigation through the experiment;
- filtering by **Gaussian**, **Fock**, **interferometer**, and **measurement** layers;
- an in-page full-resolution plot viewer, so figures can be inspected without downloading an archive;
- experiment metrics and reproducibility metadata;
- direct links to the source commit and CI run;
- raw journal files alongside the figures;
- a persistent archive of earlier commit-addressed runs.
</details>

**[→ Open the Catsy simulation explorer](https://catsy-1d3a5f.idmpages.uni-h.de/)**  <sub>also on [GitHub](https://raiyiz.github.io/catsy/)</sub>

The explorer is deliberately static: no server or Python environment is required to browse the results. The plots themselves are generated exclusively through Catsy's visualization helpers.

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

The [documentation](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst) contains the more detailed architectural and mathematical specifications.

The repository also contains examples and tests that can be useful when exploring particular operations.

## Status

`catsy` is focused on continuous-variable quantum optics and Fock states, rather than providing a general-purpose quantum-computing framework. The API is evolving, and core conventions and numerical operations are covered in the test suite.


## Test coverage

Coverage is collected with [`pytest-cov`](https://pytest-cov.readthedocs.io/) and is intentionally configured as a development concern rather than a runtime dependency. Branch coverage (not just line coverage) is enabled via `[tool.coverage.run]` in `pyproject.toml`. Locally, run:

If the development dependencies have changed, refresh the lockfile first with `uv lock`. Then run:

```bash
uv sync --group dev
uv run pytest --cov=src/catsy --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml
```

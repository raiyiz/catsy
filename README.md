
# catsy

`catsy` is a Python toolkit for continuous-variable quantum optics, built around Gaussian states in phase space.

The main idea is to keep Gaussian calculations in phase space where possible, describe experiments as reusable circuits, and move to a truncated Fock-space representation when needed.


### For a more detailed documentation, check the latest build of the [Specs (PDF)](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst)

---

***⚠ ATTENTION! This package was build with heavy use of AI***

---


## Quick start

```python
from catsy import GaussianCircuit, GaussianOperations

initial = GaussianOperations.create_epr_pair("a", "b", r=0.7)

circuit = (
    GaussianCircuit()
    .add_mode("a")
    .add_mode("b")
    .loss("a", eta=0.9)
)

final = circuit.compile_and_run(initial_state=initial)
final.plot_covariance()
```

Here `r` is the squeezing strength and `eta` is the power transmissivity of the loss channel.

## Where to start

| If you want to...                           | Use                               |
| ------------------------------------------- | --------------------------------- |
| Create Gaussian states                      | `GaussianOperations`              |
| Apply Gaussian operations                   | `GaussianOperations`              |
| Build a sequence of operations              | `GaussianCircuit`                 |
| Model loss and thermal noise                | `LossChannels`, `GaussianChannel` |
| Perform homodyne or heterodyne measurements | `GaussianMeasurements`            |
| Inspect a covariance matrix                 | `GaussianState`                   |
| Calculate a Wigner function                 | `compute_wigner_analytically()`   |
| Convert to Fock space                       | `GaussianState.to_qutip()`        |
| Define an optical layout                    | `OpticalSetup`                    |
| Save states and experiments                 | `SimulationJournal`               |

## Gaussian states

States are represented in phase space by their first moments and covariance matrix. Common operations include:

* vacuum, coherent, and EPR states
* squeezing and displacement
* phase shifts and beam splitters
* loss and thermal channels
* homodyne and heterodyne measurements

For example:

```python
from catsy import GaussianOperations

state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(state, "a", r=0.5)
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

## Circuits

For a sequence of operations that you want to keep as a reusable experiment, `GaussianCircuit` provides an explicit representation:

```python
from catsy import GaussianCircuit, GaussianOperations
import numpy as np

initial = GaussianOperations.create_vacuum(("a", "b"))

circuit = (
    GaussianCircuit()
    .add_mode("a")
    .add_mode("b")
    .squeeze("a", r=0.7)
    .squeeze("b", r=0.7, theta=np.pi / 2)
    .beam_splitter("a", "b", eta=0.5)
    .loss("a", eta=0.9)
)

final = circuit.compile_and_run(initial_state=initial)
```

Circuits can also be serialized and restored.

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
git clone https://github.com/raiyiz/catsy.git
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

```text
src/catsy/
├── core.py       conventions, validation, numerical helpers
├── gaussian.py   states, operations, channels, circuits, measurements
├── fock.py       Fock-space functionality
├── optics.py     optical layouts and QuTiP-based cavity/interferometer
│                 simulations
└── journal.py    experiment persistence
```

## Documentation

The [documentation](https://gitlab.uni-hannover.de/afam/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst) contains the more detailed API and usage information.

The repository also contains examples and tests that can be useful when exploring particular operations.

## Status

`catsy` is focused on continuous-variable quantum optics rather than providing a general-purpose quantum-computing framework. The API is still developing, but the core conventions and numerical operations are covered by the test suite.


<sub>cats & states & oha & phos & nothingness, very pur so</sub>

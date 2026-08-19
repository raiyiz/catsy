
# catsy

`catsy` is a Python toolkit for continuous-variable quantum optics, built around Gaussian states in phase space.

The main idea is to keep Gaussian calculations in phase space where possible, describe experiments as reusable circuits, and move to a truncated Fock-space representation when needed.


### For a more detailed documentation, check the latest build of the [Specs (PDF)](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst)


### Status

[![Pipeline status](https://gitlab.uni-hannover.de/inl/catsy/badges/main/pipeline.svg)](https://gitlab.uni-hannover.de/inl/catsy/-/pipelines) [![Coverage](https://gitlab.uni-hannover.de/inl/catsy/badges/main/coverage.svg)](https://inl.gitlab-pages.uni-hannover.de/catsy/) [Latest HTML coverage report](https://inl.gitlab-pages.uni-hannover.de/catsy/)

The default branch publishes the latest interactive HTML coverage report through GitLab Pages. Every pipeline (including merge requests) also attaches its own HTML coverage report directly to the `pytest` job -- open the job and use the "HTML coverage report" artifact link if the default-branch Pages link above is ever out of date. The CI pipeline also uploads the Cobertura XML report so GitLab can show the coverage percentage and line-by-line coverage annotations in merge requests.

If the Pages link above 404s, GitLab Pages may not be deployed for this project yet, or `inl.gitlab-pages.uni-hannover.de` may not match the domain your GitLab admin has actually configured for this instance -- check **Deploy > Pages** in the project settings for the real URL.

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
| Create Gaussian states                      | [`GaussianOperations`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L321) |
| Apply Gaussian operations                   | [`GaussianOperations`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L321) |
| Build a sequence of operations              | [`GaussianCircuit`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L695) |
| Model loss and thermal noise                | [`LossChannels`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L555), [`GaussianChannel`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L492) |
| Perform homodyne or heterodyne measurements | [`GaussianMeasurements`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L852) |
| Inspect a covariance matrix                 | [`GaussianState`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L86) |
| Calculate a Wigner function                 | [`compute_wigner_analytically()`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L979) |
| Convert to Fock space                       | [`GaussianState.to_qutip()`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py#L166) |
| Define an optical layout                    | [`OpticalSetup`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/optics.py#L164) |
| Save states and experiments                 | [`SimulationJournal`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/journal.py#L354) |

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
| [`core.py`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/core.py)         | conventions, validation, numerical helpers                          |
| [`gaussian.py`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/gaussian.py) | states, operations, channels, circuits, measurements                |
| [`fock.py`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/fock.py)         | Fock-space functionality                                             |
| [`optics.py`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/optics.py)     | optical layouts and QuTiP-based cavity/interferometer simulations   |
| [`journal.py`](https://gitlab.uni-hannover.de/inl/catsy/-/blob/main/src/catsy/journal.py)   | experiment persistence                                               |

## Documentation

The [documentation](https://gitlab.uni-hannover.de/inl/catsy/-/jobs/artifacts/main/raw/architectural_specs.pdf?job=typst) contains the more detailed API and usage information.

The repository also contains examples and tests that can be useful when exploring particular operations.

## Status

`catsy` is focused on continuous-variable quantum optics rather than providing a general-purpose quantum-computing framework. The API is still developing, but the core conventions and numerical operations are covered by the test suite.


<sub>cats & states & oha & phos & nothingness, very pur so</sub>

## Test coverage

Coverage is collected with [`pytest-cov`](https://pytest-cov.readthedocs.io/) and is intentionally configured as a development concern rather than a runtime dependency. Locally, run:

If the development dependencies have changed, refresh the lockfile first with `uv lock`. Then run:

```bash
uv sync --group dev
uv run pytest --cov=src/catsy --cov-report=term-missing --cov-report=html:coverage/html --cov-report=xml:coverage/coverage.xml
```

This enables branch coverage and reports missing lines. The HTML report is written to `coverage/html/`; the XML report is written to `coverage/coverage.xml`.

The GitLab CI test job runs the same coverage-enabled test suite. GitLab receives the Cobertura-compatible XML report and uses it for both the coverage percentage and line-by-line coverage annotations in merge requests. The percentage is configured with the CI `coverage` keyword, while `artifacts:reports:coverage_report` provides the line annotations. urlGitLab coverage reporting documentationhttps://docs.gitlab.com/ci/testing/code_coverage/coverage_reporting/ urlGitLab coverage visualization documentationhttps://docs.gitlab.com/ci/testing/code_coverage/coverage_visualization/

There is deliberately **no minimum coverage gate yet**. This gives the project a reliable baseline first; a `--cov-fail-under` threshold can be introduced once the initial coverage level is known and the test suite has been expanded accordingly.


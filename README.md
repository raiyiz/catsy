# catsy


`catsy` is a small continuous-variable (CV) quantum-optics toolkit built around
Gaussian states in the interleaved phase-space convention `(x1, p1, x2, p2, ...)`.

The idea is simple:

- keep Gaussian experiments in phase space as long as possible;
- describe reusable experiments as circuits;
- cross into Fock space only when a calculation actually needs it.

QuTiP provides the Fock-space representation used by `catsy`.

### For a more detailed documentation, check the latest build of the [Specs](https://gitlab.uni-hannover.de/afam/catsy/-/artifacts)

---

***⚠ ATTENTION! This package was build with heavy use of AI***

---

## Quick start

Create an EPR pair, send one mode through loss, and inspect the result:

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

print(final)
final.plot_covariance()
```

For a two-mode EPR state, `r` controls the squeezing strength. Here `eta=0.9`
means that mode `a` has 10% vacuum-coupled loss.

## Choose your starting point

| If you want to... | Start with |
| --- | --- |
| Create vacuum, coherent, or EPR states | `GaussianOperations` |
| Apply individual Gaussian gates | `GaussianOperations` |
| Build a reusable experiment | `GaussianCircuit` |
| Add loss or thermal noise | `LossChannels` / `GaussianChannel` |
| Condition a state on a measurement | `GaussianMeasurements` |
| Inspect a covariance matrix | `GaussianState.plot_covariance()` |
| Compute a Gaussian Wigner function | `compute_wigner_analytically()` |
| Convert to a Fock-space density matrix | `GaussianState.to_qutip()` |
| Describe a reusable optical layout | `OpticalSetup` |
| Persist experiment results | `SimulationJournal` |

## A few recipes

### Squeezing

```python
from catsy import GaussianOperations

state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(state, "a", r=0.5)
```

At `theta=0`, squeezing reduces `Var(x)` by `exp(-2r)` and increases
`Var(p)` by `exp(2r)`.

### Displacement

```python
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

The amplitude convention is

`alpha = (x + i p) / sqrt(2)`.

Displacement changes the first moments but leaves the covariance unchanged.

### Beam splitter

```python
state = GaussianOperations.apply_beam_splitter(
    state,
    mode_a="a",
    mode_b="b",
    eta=0.5,
)
```

`eta` is the power transmissivity. `eta=0.5` is a balanced beam splitter.

### Loss

```python
state = GaussianOperations.apply_loss(state, "a", eta=0.9)
```

This is vacuum-coupled loss. For a thermal environment, use:

```python
from catsy import LossChannels

channel = LossChannels.thermal_loss(
    mode="a",
    eta=0.9,
    n_thermal=0.2,
)
state = channel.apply(state)
```

### Homodyne measurement

```python
from catsy import GaussianMeasurements

outcome, remaining = GaussianMeasurements.homodyne_measurement(
    state,
    measured_mode="a",
    phi=0.0,
    outcome=0.2,
)
```

Passing an explicit `outcome` is useful for deterministic examples and tests.
Omit it to sample a measurement result.

### Heterodyne measurement

```python
outcome, remaining = GaussianMeasurements.heterodyne_measurement(
    state,
    measured_mode="a",
    outcome=[0.2, -0.1],
)
```

The outcome is a two-vector `(x, p)`.

### Gaussian to Fock space

```python
rho = state.to_qutip(N_cutoff=20)
```

The Gaussian state itself is represented exactly by `(d, V)`, but the returned
QuTiP density matrix is truncated. Increase `N_cutoff` until the quantity you
care about has converged.

## Circuits

For exploratory work, direct operations are convenient:

```python
state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(state, "a", r=0.5)
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

For a reproducible experiment, describe the sequence explicitly:

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

A circuit stores the recipe, then `compile_and_run()` executes it in order.
If no initial state is supplied, `add_mode(..., alpha=...)` can define the
initial coherent amplitudes.

## Conventions

`catsy` uses:

- `hbar = 1` and `[x, p] = i`;
- vacuum covariance `V_vac = I / 2`;
- quadratures ordered as `(x1, p1, x2, p2, ...)`;
- covariance matrices defined from symmetrized second moments;
- coherent amplitudes `alpha = (x + i p) / sqrt(2)`.

For a single-mode squeezed vacuum at `theta=0`:

```text
Var(x) = exp(-2r) / 2
Var(p) = exp(2r) / 2
```

These conventions are part of the API: changing them changes numerical factors
throughout the Gaussian and Fock-space layers.

## Optical layouts

`OpticalSetup` describes a reusable optical-bench layout. Execution still uses
the Gaussian layer.

```python
from catsy import OpticalSetup

setup = (
    OpticalSetup("Example interferometer")
    .beam_splitter("Input BS", "a", "b", eta=0.5)
    .inline_squeezer("Squeezer A", "a", r=0.6)
    .phase_shifter("Phase A", "a", phi=0.7853981633974483)
    .fiber_loss("Fiber", "a", eta=0.92)
)
```

Run a layout with an existing state:

```python
final = setup.process_beam(initial)
```

## Saving states and circuits

Gaussian states and circuits can be serialized:

```python
state.save("state.json")
state = GaussianState.load("state.json")

circuit.save("circuit.json")
circuit = GaussianCircuit.load("circuit.json")
```

For compact in-memory data, use `to_dict()` / `from_dict()`.

## Fock-space boundary

Most Gaussian experiments should stay in phase space. Use:

```python
rho = state.to_qutip(N_cutoff=20)
```

when you need an explicit Fock-space state. The cutoff is numerical, not
physical: increasing it should leave your observable of interest unchanged
within the desired tolerance.

## Project layout

```text
src/catsy/
├── core.py       conventions, validation, numerical helpers
├── gaussian.py   states, operations, channels, circuits, measurements, analysis
├── quantum.py    Fock-space functionality
├── optics.py     reusable optical layouts
└── journal.py    experiment persistence
```

## Testing

```bash
pytest
pytest --plot
```

The test suite checks physical invariants, analytic reference values,
serialization, measurements, and the Gaussian/Fock boundary.

## Scope

`catsy` is deliberately focused rather than a complete quantum-computing
framework. Its priority is readable CV quantum-optics mathematics, explicit
conventions, and small composable building blocks.

<sub>cats & states & oha & phos & nothingness, very pur so</sub>

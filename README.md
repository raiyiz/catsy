# catst

`catst` is a small continuous-variable (CV) quantum-optics toolkit built around
Gaussian states in the interleaved phase-space convention
`(x1, p1, x2, p2, ...)`.

The project is intentionally compact: the Gaussian phase-space layer handles
most circuit construction without allocating a Hilbert-space state, while
QuTiP provides the required Fock-space representation and non-Gaussian
calculations. QuTiP is an essential runtime dependency and is imported when
`catst` is loaded.

## Conventions

The package uses

- `ℏ = 1` and `[x, p] = i`;
- vacuum covariance `V_vac = I / 2`;
- displacement `alpha = (x + i p) / sqrt(2)`;
- covariance matrices defined as the symmetrized second moments;
- quadratures ordered as `(x1, p1, x2, p2, ...)`.

For a single-mode squeezed vacuum with squeezing strength `r` and `theta=0`,
`Var(x) = exp(-2r)/2` and `Var(p) = exp(2r)/2`.

The standard beam splitter uses power transmissivity `eta`, with the phase-space
map on the two target modes

```text
[x_a']   [ sqrt(eta)   sqrt(1-eta)] [x_a]
[x_b'] = [-sqrt(1-eta) sqrt(eta) ] [x_b]
```

and the same transformation for the `p` quadratures.

## What is included

### Gaussian phase space

- `GaussianState`: validated multimode `(d, V)` representation, mode reordering,
  serialization, plotting, and a bridge to QuTiP.
- `GaussianOperations`: vacuum/coherent/EPR construction plus squeezing,
  phase rotation, displacement, beam splitter, and vacuum loss.
- `GaussianChannel` and `QBSChannels`: general completely-positive Gaussian
  channels plus thermal loss and phase-jitter helpers.
- `GaussianCircuit`: a serializable operation sequence with an extensible
  operation registry.
- `GaussianMeasurements`: ideal homodyne and heterodyne conditioning.
- `analysis`: analytic Wigner functions, joint correlations, and a Duan-Simon
  entanglement witness.

### Fock space and non-Gaussian operations

- Williamson-based Gaussian-state conversion to a truncated QuTiP density matrix.
- Photon addition and subtraction.
- Cavity/Kerr-pulse and lossy Mach-Zehnder simulation helpers.

### Optical layouts and experiment records

- `OpticalSetup` composes named optical components into reusable layouts.
- `SimulationJournal` stores metadata, scalar results, Gaussian states, and large
  NumPy arrays in JSON/NPZ form.

## Typical workflow

```python
from catst import GaussianCircuit, GaussianOperations

# Build an EPR state and run it through a small Gaussian circuit.
initial = GaussianOperations.create_epr_pair("a", "b", r=0.7)
circuit = (
    GaussianCircuit()
    .add_mode("a")
    .add_mode("b")
    .loss("a", eta=0.9)
)
final = circuit.compile_and_run(initial_state=initial)
```

For direct gate-by-gate work:

```python
from catst import GaussianOperations

state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(state, "a", r=0.5)
state = GaussianOperations.apply_displacement(state, "a", alpha=0.4 + 0.2j)
```

## Module layout

The implementation is grouped into a few broad layers:

```text
catst/
├── core.py        shared numerical helpers and phase-space conventions
├── gaussian.py    Gaussian states, operations, channels, circuits,
│                  measurements, and phase-space analysis
├── quantum.py     QuTiP/Fock-space operations and numerical simulations
├── optics.py      reusable optical-bench layouts
└── journal.py     experiment persistence
```

The package does not maintain a compatibility facade yet. Imports should use
these modules directly, or the public names re-exported from `catst`.

## Testing

Run the full test suite with:

```bash
pytest
```

Plotting tests are opt-in:

```bash
pytest --plot
```

The test suite emphasizes both physical invariants and analytic reference
values: uncertainty-relation checks, symplectic transformations, exact loss
limits, covariance updates, circuit serialization, measurements, and the
Gaussian/Fock boundary.

## Scope

This is deliberately a focused toolkit rather than a full quantum-computing
framework. The priority is readable CV quantum-optics mathematics, explicit
conventions, and small composable building blocks rather than a large gate
catalog.

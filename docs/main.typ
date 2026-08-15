#import "@preview/physica:0.9.8": *

#set document(
  title: "catsy: A Continuous-Variable Gaussian Quantum-Optics Toolkit",
  author: "Project Documentation",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (
    top: 2.5cm,
    bottom: 2.5cm,
    left: 2.7cm,
    right: 2.7cm,
  ),
)

#set text(
  font: "New Computer Modern",
  size: 10.5pt,
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.1")

#let state = "state"
#let modes = "modes"
#let Var = "Var"

= catsy
#align(center)[
  #text(size: 1.35em, weight: "bold")[
    A Continuous-Variable Gaussian Quantum-Optics Toolkit
  ]

  #v(0.5em)

  A technical guide to the architecture, mathematical conventions,
  algorithms, and practical use of `catsy`.
]

#v(1em)

#align(center)[
  *Repository:* #link("https://github.com/raiyiz/catsy")[github.com/raiyiz/catsy]

  *Current documented branch:* `larger_refactor`

  *Python package version:* `0.1.0`
]

#v(1.5em)

== Abstract

`catsy` is a compact Python toolkit for continuous-variable (CV) quantum optics. Its central abstraction is a Gaussian quantum state represented directly in phase space by a displacement vector and covariance matrix rather than by an explicitly allocated Hilbert-space state.

This design makes Gaussian circuit construction inexpensive and transparent: Gaussian states, optical transformations, Gaussian channels, and Gaussian measurements can be manipulated using finite-dimensional vectors and matrices. When a calculation genuinely requires a Fock-space representation or a non-Gaussian operation, `catsy` provides a bridge to QuTiP.

The repository explicitly adopts the interleaved quadrature convention

$ (x_1, p_1, x_2, p_2, dots) $

together with the convention

$ hbar = 1, quad([x, p]) = i, V_"vac" = 1/2 I. $

These conventions are not cosmetic. They determine the numerical factors in displacement amplitudes, squeezing transformations, uncertainty checks, channel complete-positivity conditions, measurements, and the Gaussian-to-Fock conversion.

This document explains the implementation from the outside in: first the mathematical model, then the module architecture, then the principal classes and operations, and finally complete usage patterns.

= 1. Project at a glance

== 1.1 What problem does `catsy` solve?

The package targets simulations in which optical modes are naturally described by bosonic annihilation and creation operators, but where most of the experiment remains Gaussian.

The principal computational representation is


$ state = (modes, d, V), $

where:

- `modes` identifies the optical modes by name;
- $d$ is the vector of first moments, or displacements;
- $V$ is the covariance matrix of the quadratures.

For $n$ modes, both $d$ and the covariance matrix have dimensions

$ d in RR^(2n), quad(V) = 2n times 2n. $

A Gaussian operation is then represented by an affine phase-space transformation

$ d' = S d + d_0 $
$ V' = S V S^T + Y. $

For a unitary Gaussian operation, $Y = 0$. For a noisy Gaussian channel, $Y$ represents added noise.

This is the fundamental reason the package can perform many calculations without constructing a Hilbert-space tensor product. The implementation applies matrix transformations directly to the displacement and covariance data. The shared helper `_apply_gaussian_transform` in `core.py` implements exactly this update rule.

== 1.2 What the package contains

The repository is organized into four main implementation modules:

- `core.py` — numerical helpers, validation, phase-space conventions, Gaussian transformations, and Williamson decomposition;
- `gaussian.py` — Gaussian states, Gaussian operations, channels, circuits, measurements, and analysis;
- `quantum.py` — QuTiP/Fock-space functionality and non-Gaussian simulations;
- `optics.py` — reusable optical-bench layouts;
- `journal.py` — persistent experiment records and numerical result storage.

The README describes the same broad division and emphasizes that the package intentionally remains a focused toolkit rather than a complete quantum-computing framework.

== 1.3 Design philosophy

Three principles recur throughout the implementation.

*Phase-space first.*

Gaussian calculations remain in the $(d,V)$ representation for as long as possible.

*Explicit physical conventions.*

The package validates covariance matrices and channels against the relevant uncertainty and complete-positivity conditions rather than silently accepting arbitrary matrices.

*Explicit transition to Fock space.*

The Gaussian representation is not treated as an approximate substitute for a Fock-space calculation. Instead, the conversion is made explicit through `GaussianState.to_qutip`, which performs a Williamson decomposition and then constructs a truncated QuTiP density matrix.

= 2. Physical conventions

== 2.1 Units and commutation relation

The package uses

$ hbar = 1 $

and

$ [x,p] = i. $

With this choice, the vacuum state has covariance

$ V_"vac" = 1/2 I. $

This convention appears directly in `GaussianOperations.create_vacuum`, where the covariance matrix is initialized as `0.5 * I`.

The corresponding uncertainty relation is

$ V + i Omega / 2 >= 0, $

where $Omega$ is the symplectic form.

For $n$ modes, `catsy` constructs

$ Omega = diag(J, J, ..., J), $

with

$ J = mat(0, 1; -1, 0). $

The helper `_symplectic_form` in `core.py` creates this block-diagonal matrix according to the interleaved quadrature ordering.

== 2.2 Quadrature ordering

The package always orders quadratures as

$ (x_1, p_1, x_2, p_2, ..., x_n, p_n). $

Consequently, mode `k` occupies positions

$ 2k quad "and" 2k + 1 $

when the mode is counted from zero.

The method `GaussianState.get_mode_index` implements precisely this mapping:

```python
return self.modes.index(mode_name) * 2
```

This apparently small detail is one of the most important conventions in the project. Every covariance sub-block, every local gate, every measurement, and every conversion to QuTiP depends on the same ordering.

== 2.3 Complex coherent-state amplitudes

For a coherent-state amplitude $alpha$, the package uses

$ alpha = (x + i p) / sqrt(2), $

or equivalently

$ x = sqrt(2) Re(alpha), $
$ p = sqrt(2) Im(alpha). $

`GaussianOperations.apply_displacement` implements exactly these conversions.

Thus a call such as

```python
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

changes the first moments of mode `a` but leaves its covariance matrix unchanged.

= 3. Mathematical representation of Gaussian states

== 3.1 The `GaussianState` object

The central data structure is:

```python
@dataclass
class GaussianState:
    modes: tuple[str, ...]
    displacement: np.ndarray
    covariance: np.ndarray
```

The constructor immediately validates the state. It checks:

- mode names are unique;
- the displacement has dimension $2n$;
- the covariance matrix has shape $(2n,2n)$;
- all numerical values are finite;
- the covariance matrix is symmetric;
- the uncertainty relation is satisfied.

These checks are performed by `GaussianState._validate` and the validation functions in `core.py`.

This means that a `GaussianState` is more than a passive container. Constructing one establishes an invariant: the object represents a physically admissible Gaussian covariance matrix according to the package's convention.

== 3.2 First moments

The displacement vector is

$ d = (〈x_1〉, 〈p_1〉, 〈x_2〉, 〈p_2〉, ..., 〈x_n〉, 〈p_n〉). $

The covariance matrix stores symmetrized second moments. Informally,

$ V_"ij" = 1/2 〈Delta R_i Delta R_j + Delta R_j Delta R_i〉, $

where

$ Delta R = R - 〈R〉. $

The implementation therefore separates classical location in phase space from fluctuations around that location.

This distinction is important for operations such as displacement: displacement modifies $d$ but does not modify $V$. The implementation explicitly follows this rule.

== 3.3 Physicality validation

The physicality test is implemented as

$ V + i Omega / 2 >= 0. $

The code evaluates the smallest eigenvalue of this Hermitian matrix and rejects the covariance matrix if the eigenvalue is below the configured numerical tolerance.

This is preferable to checking only that $V$ is positive definite. A positive covariance matrix is necessary but, in quantum mechanics, not sufficient.

For example, a matrix may be numerically positive but still violate the canonical uncertainty relation. `catsy` catches this distinction at the state boundary.

= 4. Gaussian operations

== 4.1 General transformation rule

The package centralizes Gaussian state updates in `_apply_gaussian_transform`.

Given a transformation matrix $S$, optional noise matrix $Y$, and optional affine displacement $d_0$, the update is

$ d' = S d + d_0 $
$ V' = S V S^T + Y. $

This abstraction allows unitary Gaussian gates and noisy Gaussian channels to share the same numerical foundation.

== 4.2 Vacuum states

A multimode vacuum is created with:

```python
state = GaussianOperations.create_vacuum(("a", "b"))
```

The resulting state has

$ d = 0 $

and

$ V = 1/2 I_4. $

The implementation creates the covariance using `0.5 * np.eye(dim)`.

== 4.3 Coherent states

A coherent state is constructed by starting from vacuum and applying a displacement to each mode.

```python
state = GaussianOperations.create_coherent(
    ("a", "b"),
    [0.4 + 0.2j, -0.1 + 0.3j],
)
```

A scalar amplitude can be broadcast to every mode.

Because coherent states have the vacuum covariance, their covariance remains

$ V = 1/2 I $

while their first moments change.

== 4.4 Squeezing

Single-mode squeezing is implemented through

$ S_"local" =
  R(theta)
  mat(e^(-r), 0; 0, e^r)
  R(theta)^T. $

The implementation embeds this $2 times 2$ transformation into the full multimode identity matrix and applies it to the selected mode.

For `theta = 0`, this gives

$ x -> e^(-r) x, $
$ p -> e^r p. $

Therefore a vacuum mode obtains

$ Var(x) = e^(-2r)/2, $
$ Var(p) = e^(2r)/2. $

This is the convention documented by the repository itself.

Example:

```python
state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(
    state,
    mode="a",
    r=0.5,
)
```

== 4.5 Phase rotations

A phase rotation uses

$ R(phi) =
  mat(
    cos(phi), -sin(phi);
    sin(phi),  cos(phi)
  ). $

It is applied to the selected mode without changing the covariance eigenvalues or introducing noise. The implementation describes this as a passive, energy-preserving Gaussian gate.

Example:

```python
state = GaussianOperations.apply_phase_rotation(
    state,
    mode="a",
    phi=np.pi / 4,
)
```

== 4.6 Displacement

Displacement is an affine operation rather than a symplectic mixing of quadratures.

For a complex amplitude $alpha$,

$ Delta x = sqrt(2) Re(alpha), $
$ Delta p = sqrt(2) Im(alpha). $

The implementation accepts either `alpha` or the direct pair `(x, p)`, and explicitly rejects supplying both forms simultaneously.

Example:

```python
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

The covariance is copied unchanged.

== 4.7 Beam splitters

The standard two-mode beam splitter uses power transmissivity $eta$ and transforms the two target modes according to

$ mat(x_a'; x_b')
 =
 mat(
   sqrt(eta), sqrt(1-eta);
   -sqrt(1-eta), sqrt(eta)
 )
 mat(x_a; x_b), $

with the same transformation applied to the $p$ quadratures.

The implementation validates

$ 0 <= eta <= 1 $

before constructing the corresponding global transformation.

A balanced beam splitter is therefore simply

```python
state = GaussianOperations.apply_beam_splitter(
    state,
    mode_a="a",
    mode_b="b",
    eta=0.5,
)
```

= 5. Gaussian entanglement and EPR states

== 5.1 EPR-pair construction

`GaussianOperations.create_epr_pair` creates a canonical two-mode squeezed state by:

+ creating two vacuum modes;
+ squeezing mode `a` in the $x$ direction;
+ squeezing mode `b` in the $p$ direction;
+ mixing them on a 50:50 beam splitter.

The implementation documents the resulting correlations as

$ Var(x_a - x_b) = Var(p_a + p_b) = e^(-2r). $

For positive $r$, these joint quadrature variances are reduced below the vacuum level.

Example:

```python
epr = GaussianOperations.create_epr_pair(
    "a",
    "b",
    r=0.7,
)
```

== 5.2 Why the implementation builds the EPR state this way

The construction is physically transparent.

A single-mode squeezed state has reduced fluctuations in one quadrature and increased fluctuations in the conjugate quadrature. Combining two appropriately oriented squeezed states on a balanced beam splitter converts these local squeezing resources into nonlocal correlations.

This also makes the routine useful as a reference construction: the resulting covariance matrix can be used to test beam splitters, squeezing, measurements, entanglement diagnostics, and Fock-space conversion.

== 5.3 Duan-Simon witness

The package defines `DUAN_SEPARABILITY_BOUND = 2.0` in `core.py`, and the analysis layer provides a Duan-Simon-style inseparability calculation.

For the convention used by `catsy`, the relevant combined quadrature variances can be compared against the separability threshold.

This is useful because ordinary covariance or scatter plots can reveal correlations without proving entanglement. The repository explicitly distinguishes correlated noise from genuine CV entanglement.

= 6. Gaussian channels

== 6.1 Channel representation

A general Gaussian channel is represented by three objects:

$ (X, Y, d_0). $

Its action is

$ d' = X d + d_0 $
$ V' = X V X^T + Y. $

For a channel to be physically valid, the noise matrix must satisfy the Gaussian complete-positivity condition

$ Y + i/2 (Omega - X Omega X^T) >= 0. $

`_validate_gaussian_channel` checks the dimensions, finiteness, symmetry of $Y$, and the minimum eigenvalue of this complete-positivity matrix.

== 6.2 Thermal loss

`LossChannels.thermal_loss` models loss with transmissivity $eta$ and thermal occupation $n_"thermal"$.

The implementation constructs

$ X = sqrt(eta) I $

and

$ Y = (1-eta)(n_"thermal" + 1/2) I. $

Thus the environment contributes the covariance of a thermal mode scaled by the lost fraction.

Example:

```python
channel = LossChannels.thermal_loss(
    mode="a",
    eta=0.9,
    n_thermal=0.2,
)
```

== 6.3 Phase jitter

The package includes a classical phase-jitter approximation in which additional noise is added to the $p$ quadrature.

The channel is constructed as

$ X = I, $
$ Y = mat(0, 0; 0, sigma_phi^2). $

This is explicitly described in the implementation as a small-angle approximation.

== 6.4 Correlated thermal noise

`LossChannels.correlated_thermal_noise` represents two modes coupled to the same noisy environment.

The cross-mode environment covariance is parameterized by `c_correlation`, and the implementation requires

$ |c_"correlation"| <= n_"thermal". $

The resulting noise matrix has diagonal thermal blocks and correlated cross blocks.

This distinction is particularly useful in simulations where two modes experience a common environment rather than independent attenuation.

= 7. Circuits

== 7.1 Why use `GaussianCircuit`?

Direct gate-by-gate manipulation is convenient for exploratory calculations:

```python
state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(state, "a", r=0.5)
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

For reproducible experiments, however, it is often better to represent the experiment itself as a sequence of operations.

`GaussianCircuit` provides that abstraction.

The circuit registers named modes and stores operations in order. `compile_and_run` subsequently validates the operation sequence and executes it sequentially.

== 7.2 Constructing a circuit

A typical circuit is:

```python
circuit = (
    GaussianCircuit()
    .add_mode("a")
    .add_mode("b")
    .squeeze("a", r=0.7)
    .squeeze("b", r=0.7, theta=np.pi / 2)
    .beam_splitter("a", "b", eta=0.5)
    .loss("a", eta=0.9)
)
```

The circuit itself stores the recipe. It does not need to contain a Hilbert-space state.

== 7.3 Running a circuit

An initial state may be supplied explicitly:

```python
initial = GaussianOperations.create_vacuum(("a", "b"))

final = circuit.compile_and_run(
    initial_state=initial,
)
```

If no initial state is supplied, the circuit constructs a coherent state from its configured initial amplitudes.

Before execution, the implementation verifies that:

- at least one mode has been registered;
- the initial state's mode set agrees with the circuit;
- an initial state is reordered into the circuit's canonical mode order;
- every operation refers only to registered modes;
- every operation name exists in the operation registry.

The registry then dispatches the operation to its implementation.

== 7.4 The operation registry

The registry is an important extensibility point.

Conceptually, execution is

```python
current_state = OPERATION_REGISTRY[op.name](
    current_state,
    op.modes,
    **op.kwargs,
)
```

This separates the *description* of an operation from its *execution*. A circuit therefore behaves more like a small intermediate representation than a collection of already-executed matrix multiplications.

That separation is also what makes serialization practical.

= 8. Measurements

== 8.1 Homodyne measurement

The Gaussian measurement layer supports ideal homodyne conditioning.

A homodyne measurement selects a quadrature of a mode and conditions the remaining Gaussian state on the observed result. The resulting state is again Gaussian, with its first moments and covariance updated by Gaussian conditioning formulas.

The important conceptual point is that a measurement is not represented as another deterministic symplectic gate. It is a probabilistic state-update operation.

== 8.2 Heterodyne measurement

The implementation provides `heterodyne_measurement`, which measures both quadratures simultaneously.

The code models heterodyne detection through an effective covariance that includes an additional vacuum contribution. The outcome is a two-vector

$ (x, p) $

sampled from the corresponding multivariate Gaussian distribution when an explicit outcome is not supplied.

Example:

```python
outcome, remaining = GaussianMeasurements.heterodyne_measurement(
    state,
    measured_mode="a",
)
```

Supplying an explicit outcome is useful for deterministic tests:

```python
outcome, remaining = GaussianMeasurements.heterodyne_measurement(
    state,
    measured_mode="a",
    outcome=np.array([0.2, -0.1]),
)
```

The implementation validates that an explicit heterodyne outcome has shape `(2,)` and contains finite values.

= 9. The Gaussian-to-Fock boundary

== 9.1 Why a boundary is necessary

Gaussian phase-space methods are extremely efficient for Gaussian circuits, but they cannot by themselves represent arbitrary non-Gaussian quantum states.

`catsy` therefore provides `GaussianState.to_qutip`, which converts the Gaussian state into a truncated QuTiP density matrix.

The implementation makes the approximation explicit: the mathematical Gaussian state is represented exactly by its covariance data, but the returned Fock-space density matrix is finite-dimensional and therefore subject to cutoff error.

== 9.2 Williamson decomposition

The conversion starts with a Williamson decomposition

$ V = S D S^T, $

where

$ D = diag(nu_1, nu_1, nu_2, nu_2, ..., nu_n, nu_n). $

The $nu_i$ are the symplectic eigenvalues.

`core.py` obtains this decomposition by:

+ computing the positive square root of $V$;
+ forming $A Omega A$;
+ performing a real Schur decomposition;
+ extracting the symplectic eigenvalues;
+ constructing the symplectic matrix $S$;
+ checking both the symplectic and covariance reconstruction residuals.

The implementation explicitly rejects numerically singular cases and raises an error if the reconstructed matrices do not meet the configured numerical tolerance.

== 9.3 Thermal Williamson modes

Once the covariance is diagonalized, each Williamson mode is represented by a thermal state.

The implementation creates these using

```python
qt.thermal_dm(
    N_cutoff,
    max(float(nu) - 0.5, 0.0),
)
```

and tensors the resulting density matrices together.

This follows from the fact that, under the chosen $hbar=1$ convention, a thermal state's covariance scale is

$ nu = n + 1/2. $

== 9.4 Symplectic polar decomposition

The implementation decomposes the Williamson symplectic transformation as

$ S = P O, $

where $P$ is positive symplectic and $O$ is orthogonal symplectic.

The passive component $O$ is implemented as a number-conserving quadratic unitary, while the positive component $P$ is implemented using a quadratic Hamiltonian constructed from `log(P)`.

The displacement is finally implemented with QuTiP's displacement primitive using the same

$ alpha = (x + i p)/sqrt(2) $

convention as the phase-space representation.

== 9.5 Choosing `N_cutoff`

The conversion accepts

```python
rho = state.to_qutip(N_cutoff=20)
```

The cutoff is a numerical parameter rather than a physical property of the state.

Too small a cutoff truncates significant Fock populations. Increasing the cutoff improves the representation but increases the size of the Hilbert space exponentially with the number of modes.

Consequently, a practical workflow is to repeat the calculation with progressively larger cutoffs and verify that the observable of interest has converged.

= 10. Optical layouts

== 10.1 Motivation

`GaussianCircuit` describes an executable sequence of operations.

`OpticalSetup`, by contrast, describes a reusable optical-bench layout.

The distinction is intentional: an optical layout is a structural object, while Gaussian execution remains the responsibility of the Gaussian layer. `optics.py` explicitly states that numerical and physical execution belongs to `GaussianOperations`, while the component table defines the structural contract for a layout component.

== 10.2 Optical components

An `OpticalComponent` contains:

- a human-readable `name`;
- an operation type;
- the ordered ports it connects;
- operation parameters.

The supported component vocabulary currently includes:

- `BeamSplitter`;
- `Loss`;
- `Squeezing`;
- `PhaseRotation`.

The component type determines how many ports are required and which parameters are valid.

== 10.3 Building a setup

For example:

```python
setup = (
    OpticalSetup("Example interferometer")
    .beam_splitter("Input BS", "a", "b", eta=0.5)
    .inline_squeezer("Squeezer A", "a", r=0.6)
    .phase_shifter("Phase A", "a", phi=np.pi / 4)
    .fiber_loss("Fiber", "a", eta=0.92)
)
```

The convenience methods are thin wrappers around `add_component`. The implementation currently provides dedicated builders for beam splitters, fiber loss, inline squeezing, and phase shifting.

== 10.4 Mapping layouts to circuits

The optical layer maintains a mapping from component type to the corresponding `GaussianCircuit` builder:

```python
_COMPONENT -> GaussianCircuit operation
```

For example, a `BeamSplitter` component is mapped to `circuit.beam_splitter`, while `Loss`, `Squeezing`, and `PhaseRotation` map to their corresponding circuit methods.

This means that adding a new optical component should normally involve extending the component vocabulary and its builder mapping rather than rewriting the general processing mechanism.

= 11. Experiment journals

== 11.1 Purpose

Scientific simulations frequently produce much more numerical data than should be loaded merely to inspect experiment metadata.

`SimulationJournal` addresses this by splitting each saved entry into:

- a JSON metadata file;
- an optional compressed NPZ array file.

The JSON stores titles, tags, notes, scalar results, and array metadata, while large NumPy arrays are kept in the companion NPZ file.

== 11.2 Why JSON and NPZ are separated

A large numerical array is inefficient in JSON because its values become decimal text. Moreover, loading a JSON file requires parsing the entire document even if the caller only wants its title.

The journal therefore stores heavy arrays separately. The implementation notes that NumPy's NPZ loading is lazy with respect to individual arrays, allowing callers to avoid paying the cost of arrays they never request.

The resulting layout is conceptually:

```text
journal/
├── entry_<id>.json
└── entry_<id>.npz
```

The NPZ file is only created when an entry contains array data.

== 11.3 Logging a run

A run can contain:

- its name;
- the `GaussianCircuit`;
- scalar metrics;
- a final `GaussianState`;
- arbitrary numerical arrays.

Example:

```python
journal_entry.log_run(
    "loss sweep",
    circuit=circuit,
    final_state=final,
    metrics={
        "purity": 0.98,
    },
    arrays={
        "wigner_grid": wigner_values,
    },
)
```

The implementation stores the circuit using `to_dict`, while the final state's displacement and covariance become separate NPZ payloads.

== 11.4 Array annotations

Large arrays may additionally be accompanied by metadata such as:

- description;
- unit;
- dimensions;
- shape;
- dtype.

This allows a journal entry to remain searchable and inspectable without loading the numerical payload itself.

== 11.5 Atomic persistence

The journal's design notes specify atomic saves: data are written to temporary files and then moved into place with `Path.replace`.

The purpose is simple: if a process crashes during a save, an already-existing journal entry should not be left partially overwritten.

= 12. Serialization

== 12.1 Gaussian states

A `GaussianState` can be converted into a plain dictionary:

```python
data = state.to_dict()
```

The dictionary contains:

```text
modes
displacement
covariance
```

It can then be reconstructed with:

```python
state = GaussianState.from_dict(data)
```

or persisted directly with:

```python
state.save("state.json")
```

and restored with:

```python
state = GaussianState.load("state.json")
```

The implementation uses JSON for these compact state representations.

== 12.2 Gaussian channels

`GaussianChannel` follows the same pattern, storing:

- target modes;
- `X`;
- `Y`;
- `d0`.

Channels can therefore also be serialized and restored.

== 12.3 Circuits

Circuit operations are represented structurally and can be converted to dictionaries. This is particularly useful for experiment journals, because the journal can preserve the recipe used to generate a numerical result rather than only preserving the final state.

= 13. Analysis and visualization

The Gaussian module also contains phase-space analysis utilities. The repository identifies analytic Wigner functions, joint correlations, and a Duan-Simon entanglement witness as part of the analysis layer.

`GaussianState.plot_covariance` visualizes the full covariance matrix and labels the axes with the corresponding quadrature names, such as `q_a`, `p_a`, `q_b`, and `p_b`.

Example:

```python
state.plot_covariance()
```

The plotting layer is deliberately secondary to the numerical representation: plotting reads the already-computed state rather than changing the state itself.

= 14. A complete example

The following example illustrates the intended high-level workflow.

```python
from catsy import GaussianCircuit, GaussianOperations

# 1. Prepare an entangled Gaussian input state.
initial = GaussianOperations.create_epr_pair(
    "a",
    "b",
    r=0.7,
)

# 2. Describe the experiment as a circuit.
circuit = (
    GaussianCircuit()
    .add_mode("a")
    .add_mode("b")
    .loss("a", eta=0.9)
    .rotate("b", phi=0.2)
    .beam_splitter("a", "b", eta=0.5)
)

# 3. Execute it in phase space.
final = circuit.compile_and_run(
    initial_state=initial,
)

# 4. Inspect the resulting Gaussian state.
print(final)

# 5. Plot its covariance matrix.
final.plot_covariance()

# 6. Convert to a truncated Fock-space density matrix if required.
rho = final.to_qutip(N_cutoff=20)
```

The important architectural point is the order of operations:

$"Gaussian preparation"
 "Gaussian circuit"
 "Gaussian analysis"
 "optional Fock conversion". $

Most experiments should remain on the left side of this boundary until a genuinely non-Gaussian calculation is required.

= 15. Direct operations versus circuits

There are two equally legitimate styles.

== 15.1 Direct functional style

Use direct operations when exploring a small calculation interactively.

```python
state = GaussianOperations.create_vacuum(("a",))
state = GaussianOperations.apply_squeezing(
    state,
    "a",
    r=0.5,
)
state = GaussianOperations.apply_displacement(
    state,
    "a",
    alpha=0.4 + 0.2j,
)
```

This style makes each transformation explicit.

== 15.2 Circuit style

Use `GaussianCircuit` when the sequence itself is part of the experiment.

```python
circuit = (
    GaussianCircuit()
    .add_mode("a")
    .squeeze("a", r=0.5)
    .displace("a", alpha=0.4 + 0.2j)
)

state = circuit.compile_and_run()
```

The circuit style has advantages for reproducibility, serialization, experiment journals, and reusable optical layouts.

= 16. Validation and failure modes

`catsy` generally prefers explicit validation over silently repairing invalid input.

Some important examples are:

- transmissivities must lie in $[0,1]$;
- thermal occupation numbers must be non-negative;
- mode names must be unique;
- covariance matrices must have even dimension;
- covariance matrices must be symmetric;
- covariance matrices must satisfy the Gaussian uncertainty relation;
- Gaussian channel matrices must have compatible dimensions;
- Gaussian channel noise must satisfy complete positivity;
- correlated thermal noise must satisfy its physical correlation bound;
- heterodyne outcomes must be finite two-vectors;
- circuit operations must reference registered modes.

These checks are implemented centrally in `core.py` and at the relevant public API boundaries.

This has an important practical consequence: if a calculation fails, the exception often identifies a violated physical or structural invariant rather than producing an apparently plausible but invalid state.

= 17. Numerical tolerances

`core.py` defines several numerical thresholds:

```python
TOL_ZERO_ENTRY = 1e-9
TOL_TRACE_WARN = 1e-6
TOL_PHYSICALITY = 1e-10
DUAN_SEPARABILITY_BOUND = 2.0
```

The physicality tolerance is used when testing uncertainty relations, Gaussian-channel complete positivity, and related numerical conditions.

These values should be interpreted as numerical tolerances, not physical constants.

When extending the package, avoid introducing ad-hoc tolerances unless there is a specific numerical reason. Reusing the existing tolerance policy helps keep validation behavior consistent across the codebase.

= 18. Testing philosophy

The repository includes a `tests` directory and recommends running:

```text
pytest
```

Plotting tests are opt-in:

```text
pytest --plot
```

The project describes its test suite as emphasizing both physical invariants and analytic reference values, including uncertainty relations, symplectic transformations, loss limits, covariance updates, circuit serialization, measurements, and the Gaussian/Fock boundary.

For contributors, this means that a new operation should ideally be tested at two levels:

+ *structural correctness:* shapes, mode selection, serialization, and invalid inputs;
+ *physical correctness:* known covariance transformations, limiting cases, and relevant invariants.

For example, a new passive Gaussian operation should preserve the canonical symplectic form, while a pure unitary Gaussian operation should not add covariance noise.

= 19. Dependency and installation model

The project metadata declares Python `>=3.13` and depends on NumPy, SciPy-related functionality through the implementation environment, Matplotlib, QuTiP, IPython, pytest, and isort. The package uses `uv_build` as its build backend and exposes a `catsy` console script.

The repository currently declares:

```toml
[project]
name = "catsy"
version = "0.1.0"
requires-python = ">=3.13"
```

and includes:

```toml
[project.scripts]
catsy = "catsy:main"
```

The important runtime dependency for the Fock-space bridge is QuTiP. The Gaussian module imports QuTiP directly, and the repository explicitly describes QuTiP as an essential runtime dependency.

A typical development workflow is therefore:

```text
git clone https://github.com/raiyiz/catsy
cd catsy
uv sync
pytest
```

The exact environment-management commands may vary with the development setup, but the repository's Python metadata establishes Python 3.13 or newer as the intended runtime.

= 20. Recommended mental model for users

The easiest way to understand `catsy` is to treat it as a stack.

#align(center)[
  #box(
    width: 90%,
    inset: 1em,
    stroke: 0.7pt,
  )[
    *Physical conventions*

    $ hbar = 1 $, interleaved $(x,p)$, vacuum covariance $I/2$

    #v(0.5em)

    ↓

    *Gaussian state*

    $(modes, d, V)$

    #v(0.5em)

    ↓

    *Gaussian operations*

    squeezing · rotation · displacement · beam splitter

    #v(0.5em)

    ↓

    *Gaussian channels*

    loss · thermal noise · phase jitter · correlated noise

    #v(0.5em)

    ↓

    *Circuit / optical layout*

    reusable experiment description

    #v(0.5em)

    ↓

    *Measurements and analysis*

    conditioning · Wigner functions · correlations · entanglement

    #v(0.5em)

    ↓

    *Fock-space boundary*

    Williamson decomposition → QuTiP density matrix

    #v(0.5em)

    ↓

    *Experiment journal*

    metadata + scalar results + compressed numerical arrays
  ]
]

This layered model explains most of the repository.

The `core` layer defines the mathematical rules and validation.

The `gaussian` layer turns those rules into states, gates, channels, circuits, and measurements.

The `optics` layer gives the same operations an experiment-oriented structural representation.

The `quantum` layer handles calculations that need explicit Fock-space states.

The `journal` layer makes the resulting experiments persistent and searchable.

= 21. Extending `catsy`

== 21.1 Adding a Gaussian operation

A new Gaussian operation should normally be formulated as an affine phase-space transformation.

Given

$ d' = S d + d_0, $
$ V' = S V S^T + Y, $

the implementation can reuse `_apply_gaussian_transform`.

For a unitary Gaussian gate:

$ Y = 0. $

For a noisy Gaussian process:

$ Y >= 0 $

must be supplemented by the appropriate complete-positivity condition.

The new operation should then be exposed through `GaussianOperations` and, if appropriate, through the `GaussianCircuit` registry.

== 21.2 Adding a channel

A new channel should provide `X`, `Y`, and `d0`, and should pass `_validate_gaussian_channel`.

This is preferable to implementing the same physical validity checks independently in every channel factory because the common validator guarantees a consistent definition of Gaussian complete positivity.

== 21.3 Adding an optical component

The optical layer intentionally separates component description from numerical execution.

To extend the component vocabulary:

+ add the structural contract to `_COMPONENT_SPECS`;
+ add a circuit builder to `_CIRCUIT_BUILDERS`;
+ add the corresponding convenience method to `OpticalSetup` if useful;
+ add a suitable schematic abbreviation if it should appear in rendered layouts.

The existing code comments explicitly recommend extending this mapping rather than modifying the general beam-processing function.

== 21.4 Adding persistent data

For journal data, prefer scalar values for small metrics and arrays for heavy numerical payloads.

A final Gaussian state's displacement and covariance should generally be logged through the `final_state` argument rather than duplicated manually as arbitrary arrays, because the journal already knows how to store and reconstruct those fields.

= 22. Common mistakes

== 22.1 Mixing quadrature conventions

Do not silently switch between

$ (x_1,p_1,x_2,p_2) $

and

$ (x_1,x_2,p_1,p_2). $

`catsy` uses the first convention throughout its Gaussian representation.

== 22.2 Mixing $hbar$ conventions

A common source of factor-of-two errors is importing formulas written for a different value of $hbar$.

For `catsy`,

$ [x,p] = i $

and

$ V_"vac" = I/2. $

Any formula implemented outside the package should first be converted into this convention.

== 22.3 Treating displacement as a covariance transformation

Displacement changes the first moments:

$ d -> d + d_0 $

but does not change $V$.

The implementation follows this exactly.

== 22.4 Treating correlated noise as entanglement

Classical or environmental correlations can produce visibly correlated quadratures without generating quantum entanglement.

For this reason, the package includes an entanglement witness rather than relying on visual covariance inspection alone.

== 22.5 Using a low Fock cutoff

A successful call to `to_qutip` does not prove that the chosen cutoff is sufficient.

Always check convergence when observables depend significantly on high photon numbers.

== 22.6 Assuming a state can be reordered arbitrarily without transforming its covariance

`GaussianState.reorder_modes` correctly permutes both the displacement and covariance matrix.

When implementing equivalent functionality elsewhere, the same permutation must be applied to both axes of the covariance matrix. The repository deliberately treats mode order as representation metadata rather than physical state identity.

= 23. Reproducibility recommendations

For a serious numerical experiment, store at least:

+ the initial state or its preparation circuit;
+ all channel parameters;
+ the circuit operation sequence;
+ the final Gaussian state;
+ scalar observables;
+ numerical grids or other heavy arrays needed to reproduce plots;
+ the Fock cutoff when a Gaussian-to-QuTiP conversion was used.

`SimulationJournal` is designed for exactly this style of record keeping. Its separation between metadata and arrays makes experiment browsing inexpensive while retaining the numerical payload when needed.

A good experiment record should make it possible to answer:

*What state was prepared?*

*Which operations were applied?*

*With which parameters?*

*What numerical state resulted?*

*Which scalar quantities were extracted?*

*What numerical cutoff or approximation was used?*

= 24. Summary of the public conceptual API

#table(
  columns: (1.8fr, 2.5fr, 3.5fr),
  stroke: 0.5pt,
  inset: 6pt,

  [*Layer*], [*Main object*], [*Purpose*],

  [`core.py`],
  [`_symplectic_form`, validation helpers, `_apply_gaussian_transform`, Williamson decomposition],
  [Defines the mathematical and numerical foundations of the Gaussian layer.],

  [`gaussian.py`],
  [`GaussianState`],
  [Represents a physical multimode Gaussian state as `(modes, d, V)`.],

  [`gaussian.py`],
  [`GaussianOperations`],
  [Creates states and applies Gaussian gates.],

  [`gaussian.py`],
  [`GaussianChannel` / `LossChannels`],
  [Represents and constructs Gaussian noise processes.],

  [`gaussian.py`],
  [`GaussianCircuit`],
  [Stores and executes a sequence of named Gaussian operations.],

  [`gaussian.py`],
  [`GaussianMeasurements`],
  [Performs Gaussian homodyne and heterodyne conditioning.],

  [`optics.py`],
  [`OpticalComponent` / `OpticalSetup`],
  [Describes reusable optical-bench layouts.],

  [`journal.py`],
  [`JournalEntry` / `SimulationJournal`],
  [Persists experiment metadata, states, scalar metrics, and arrays.],

  [`quantum.py`],
  [QuTiP/Fock-space routines],
  [Provides explicit Hilbert-space and non-Gaussian numerical calculations.]
)

= 25. Minimal quick-reference

For a new user, the shortest path into the package is:

```python
from catsy import GaussianCircuit, GaussianOperations

# Prepare.
state = GaussianOperations.create_vacuum(("a", "b"))

# Transform.
state = GaussianOperations.apply_squeezing(
    state,
    "a",
    r=0.5,
)

state = GaussianOperations.apply_beam_splitter(
    state,
    "a",
    "b",
    eta=0.5,
)

# Analyze.
print(state)
state.plot_covariance()

# Convert only when needed.
rho = state.to_qutip(N_cutoff=20)
```

For a reusable experiment:

```python
initial = GaussianOperations.create_epr_pair(
    "a",
    "b",
    r=0.7,
)

circuit = (
    GaussianCircuit()
    .add_mode("a")
    .add_mode("b")
    .loss("a", eta=0.9)
    .beam_splitter("a", "b", eta=0.5)
)

final = circuit.compile_and_run(
    initial_state=initial,
)
```

The repository's own README presents essentially this phase-space-first workflow: construct an EPR state, register its modes in a circuit, apply loss, and execute the circuit.

= 26. References

The following references document both the implementation being described and the Typst facilities used to typeset this documentation.

#bibliography("references.bib", style: "ieee")

#import "@preview/physica:0.9.8": *

= Chapter 1: Mathematical Foundations & Phase-Space Representation

Catsy is designed for efficient simulation of continuous-variable (CV) quantum optics. The toolkit uses two complementary state representations: `GaussianState` for Gaussian states in phase space, and `FockState` for states represented in a truncated Fock space. The Gaussian layer is compact because a Gaussian state is completely specified by its first and second statistical moments. Non-Gaussian operations require the richer Fock-space representation and therefore form the transition point between the two layers.

The mathematical conventions used throughout the toolkit follow the standard Gaussian-state formalism of continuous-variable quantum information; see the reviews by Weedbrook et al. and Braunstein and van Loock for a broader treatment.
#link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and
#link("https://doi.org/10.1103/RevModPhys.77.513")[Braunstein and van Loock
(2005)] are background references for the conventions used here.

== Symplectic conventions
For a system of $n$ optical modes we define
the vector of Hermitian quadrature operators as: $r = vec(q_1, p_1, q_2, p_2, dots.v, q_n, p_n)$

The operators satisfy the canonical commutation relations (CCR), expressed symplectically as:
$[r_i, r_j] = i Omega_(i j)$

where $Omega$ is the fundamental symplectic form. Catsy implements $Omega$ as a block-diagonal matrix built from $n$ repeated $2 times 2$ blocks $J = mat(0, 1; -1, 0)$, one per mode:
$ Omega = mat(J, , dots.h; , dots.down, ; dots.v, , J) $

Using the convention $hbar = 1$, the shot-noise limit of the quantum-mechanical vacuum is defined by the covariance matrix:
$V_0 = 1/2 bb(1)_(2n)$

== Mathematical state specification (`GaussianState`)

A `GaussianState` is the toolkit's phase-space representation of a multimode Gaussian quantum state. It is fully characterized by its ordered `modes`, displacement vector $d$, and covariance matrix $V$.

1. *Displacement vector ($d in R^(2n)$):*
   $d_i = expval(r_i)_rho = "tr"(rho r_i)$

2. *Covariance matrix ($V in R^(2n times 2n)$):*
   $V_(i j) = 1/2 expval(\{r_i - d_i, r_j - d_j\})_rho$
   Positivity of the density matrix $rho >= 0$ implies the Robertson-Schrödinger uncertainty relation:
   $V + i/2 Omega >= 0$

3. *State purity:*
   The purity $mu = "Tr"(rho^2)$ is computed in phase space directly from the determinant of $V$:
   $mu = 1 / (2^n sqrt(det(V)))$
   For a pure state, strictly $det(V) = (1/2)^(2n) = 1/(4^n)$, so that $mu = 1$.

The order of `modes` is part of the representation: quadratures are stored as $(q_1,p_1,q_2,p_2,dots)$ in that order. `reorder_modes` changes this coordinate ordering while preserving the physical state. This ordering contract is used by Gaussian transformations, measurements, diagnostics, and the `Circuit` layer.

== Code architecture & validation

The public Gaussian state abstraction is `GaussianState`; its numerical invariants are enforced when the state is constructed. The implementation rejects duplicate mode names, incorrect displacement or covariance dimensions, non-finite values, non-symmetric covariance matrices, and covariance matrices that violate the physicality condition.

```python
@dataclass
class GaussianState:
    """A multi-mode Gaussian state, fully described by (modes, d, V)."""
    modes: Modes
    displacement: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        self._validate()
```

Dimensions are strictly enforced to be $2n$. A malformed input immediately raises a `ValueError`, before any subsequent Gaussian transformation can propagate an invalid matrix computation.

The physical-covariance validation and symplectic helpers are implemented in `catsy.core`, while `GaussianState` and the Gaussian-state operations are implemented in `catsy.gaussian`. The operation functions and state methods use the same mathematical transformations; the `Circuit` layer described in Chapter 3 binds those transformations to named modes as executable `Gate` instances.

== Representation boundary

The Gaussian representation is the default compact representation for states that remain Gaussian. Gaussian operations and Gaussian channels can therefore operate directly on $(d,V)$ without constructing a Fock-space density matrix.

When a computation requires a genuinely non-Gaussian operation, Catsy crosses the representation boundary with `GaussianState.to_fock()`, producing a `FockState` in a finite Fock cutoff. This is an embedding into the richer representation rather than a reversible conversion: every Gaussian state can be represented in Fock space up to truncation, but a general `FockState` cannot be reduced to a `GaussianState` without losing non-Gaussian information. Once a circuit has crossed into the Fock representation, it remains there; see Chapter 5 for the bridge and Chapter 7 for non-Gaussian operations.

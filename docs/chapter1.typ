#import "@preview/physica:0.9.8": *

= Chapter 1: Mathematical Foundations & Phase-Space Representation

This framework is optimized for the highly efficient simulation of continuous-variable (CV) quantum optics. In contrast to an explicit representation in the infinite-dimensional Fock space, the Gaussian phase-space layer uses an exact parametrization via the first and second statistical moments.

== Symplectic conventions
For a system of $n$ optical modes we define the vector of Hermitian quadrature operators as:
$r = "matrix"(q_1, p_1, q_2, p_2, dots, q_n, p_n)^T$

The operators satisfy the canonical commutation relations (CCR), expressed symplectically as:
$[r_i, r_j] = i Omega_(i j)$

where $Omega$ is the fundamental symplectic form. The toolkit implements $Omega$ as an orthogonal sum (bigoplus) of $2 times 2$ blocks across the full mode count:
 Omega = col.big(oplus, k=1, n) pmatrix(0, 1; -1, 0) 

Using the convention $hbar = 1$, the shot-noise limit of the quantum-mechanical vacuum is defined by the covariance matrix:
$V_0 = 1/2 I_(2n)$

== Mathematical state specification (`GaussianState`)
A quantum state $rho$ is fully characterized in phase space by its displacement vector $d$ and its covariance matrix $V$, provided its Wigner function is Gaussian.

1. *Displacement vector ($d in R^(2n)$):*
   $d_i = expval(r_i)_rho = "tr"(rho r_i)$

2. *Covariance matrix ($V in R^(2n times 2n)$):*
   $V_(i j) = 1/2 expval(\{r_i - d_i, r_j - d_j\})_rho$
   Positivity of the density matrix $rho >= 0$ directly implies the phase-space uncertainty relation in the form of the Robertson-Schrödinger inequality:
    V + i/2 Omega >= 0 

3. *State purity:*
   The purity $mu = "Tr"(rho^2)$ is computed in phase space directly from the determinant of $V$:
   $mu = 1 / (2^n sqrt(det(V)))$
   For a pure state, strictly $det(V) = (1/2)^(2n) = 1/(4^n)$, so that $mu = 1$.

== Code architecture & validation of `GaussianState`

The Python class `GaussianState` mirrors these invariants one-to-one. It enforces strict validation rules during initialization to rule out unphysical states at runtime.

```python
@dataclass
class GaussianState:
    """A multi-mode Gaussian state, fully described by (modes, d, V)."""
    modes: tuple[str, ...]
    displacement: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        n_modes = len(self.modes)
        if len(set(self.modes)) != n_modes:
            raise ValueError(f"Duplicate mode names in {self.modes!r}.")
        
        expected_dim = 2 * n_modes
        if self.displacement.shape != (expected_dim,):
            raise ValueError(
                f"displacement must have shape ({expected_dim},), "
                f"got {self.displacement.shape}."
            )
        if self.covariance.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"covariance must have shape ({expected_dim}, {expected_dim}), "
                f"got {self.covariance.shape}."
            )
```

Dimensions are strictly enforced to be $2n$. A malformed input immediately raises a `ValueError`, before any subsequent unitary transformation can trigger a faulty matrix computation.

---

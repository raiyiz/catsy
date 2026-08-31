#import "@preview/physica:0.9.8": *

= Chapter 2: Gaussian Unitary Transformations & Channels

Gaussian operations are transformations that preserve the Gaussian structure of the Wigner function. In catsy, the mathematical transformations are ordinary functions operating on `GaussianState`; `Gate` and `Circuit` provide the higher-level machinery for binding those transformations to named modes and sequencing them (Chapter 3). Non-unitary processes (decoherence and noise) are modeled as CPTP maps (*Completely Positive Trace-Preserving Maps*) via Gaussian channels. This is the standard Gaussian-operation/channel framework of continuous-variable quantum information; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and #link("https://doi.org/10.1103/RevModPhys.77.513")[Braunstein and van Loock (2005)].

== Gaussian transformations
Every purely Gaussian unitary transformation $hat(U)$ induces a linear transformation in phase space, expressed via a symplectic matrix $S in S_p(2n, RR)$. For the displacement vector $d$ and the covariance matrix $V$:
$d -> S d$
$V -> S V S^T$

Preserving the canonical commutation relations strictly requires that $S$ preserve the symplectic form:
$S Omega S^T = Omega$

The toolkit exposes the corresponding mathematical transformations as ordinary operations on `GaussianState`. The same operations can be bound into `Gate` instances and sequenced by `Circuit`; the state methods are therefore not the circuit abstraction itself.

1. *Squeezing operator ($hat(S)_k (r, theta)$):*
   The local squeeze operation on mode $k$ reduces the variance in one quadrature below the shot-noise limit, while amplifying the conjugate quadrature. Squeezing, passive rotations, and beam splitters are standard examples of Gaussian unitaries; their symplectic representation is developed systematically in the references above and in Serafini’s graduate-level treatment. The local symplectic matrix is:
   $ S_("local") = mat(cos(theta), -sin(theta); sin(theta), cos(theta)) mat(e^(-r), 0; 0, e^r) mat(cos(theta), sin(theta); -sin(theta), cos(theta)) $

2. *Phase rotation ($hat(R)_k (phi)$):*
   A passive, energy-preserving transformation that rotates phase space by angle $phi$:
   $ S_("local") = mat(cos(phi), -sin(phi); sin(phi), cos(phi)) $

3. *Beam splitter ($hat(B)_(k, m)(eta)$):*
   A lossless beam splitter couples modes $k$ and $m$ with power transmissivity $eta$. It can generate entanglement between independent modes:
   $t = sqrt(eta), quad r_c = sqrt(1 - eta)$
   The transformation mixes the mode pair as:
   $mat(d_k ; d_m) -> mat(t bb(1)_2, r_c bb(1)_2; -r_c bb(1)_2, t bb(1)_2) mat(d_k ; d_m)$

In the public Python API these are transformation functions such as `squeeze`, `rotate`, and `beam_splitter`. A circuit-facing call such as `circuit.squeeze("a", r=0.5)` is a builder operation: it creates a `Gate` that binds the transformation and parameters, then appends that Gate to the circuit. The mathematical operation itself does not imply circuit ownership or sequencing.

== General Gaussian channels (`GaussianChannel`)
Noise and dissipation can no longer be represented by unitary $S$-matrices alone. A general Gaussian channel is mathematically described by two real matrices $X$ and $Y$:
$d -> X d + d_0$
$V -> X V X^T + Y$

For this transformation to be a physical CPTP map, the matrix $Y$ must satisfy the noise inequality:
$Y + i/2 Omega - i/2 X Omega X^T >= 0$

The toolkit implements these maps efficiently via a global coordinate embedding in the `GaussianChannel` class:

```python
@dataclass
class GaussianChannel:
    """A general Gaussian channel d' = X@d + d0, V' = X@V@X.T + Y
    acting on a subset of modes."""
    target_modes: tuple[str, ...]
    X: np.ndarray
    Y: np.ndarray
    d0: np.ndarray

    def apply(self, state: GaussianState) -> GaussianState:
        global_dim = len(state.displacement)
        target_indices = [
            i
            for mode in self.target_modes
            for i in (state.get_mode_index(mode), state.get_mode_index(mode) + 1)
        ]
        index = np.ix_(target_indices, target_indices)

        X_global = np.eye(global_dim)
        Y_global = np.zeros((global_dim, global_dim))
        d0_global = np.zeros(global_dim)
        X_global[index] = self.X
        Y_global[index] = self.Y
        d0_global[target_indices] = self.d0

        return _apply_gaussian_transform(
            state, X_global, noise=Y_global, displacement=d0_global
        )
```

`GaussianChannel` is a plain (non-frozen) dataclass. Its `apply` method applies the channel transformation to a state; it does not represent a circuit Gate. When persistence or sequencing is required, the channel can be registered/bound as a Gate in the same way as other transformations.

== Standard optical noise channels (`LossChannels`)
The toolkit provides standard physical channels through the factory object `LossChannels`:

1. *Thermal loss channel (`thermal_loss`):*
   Models coupling to a thermal bath with mean photon number $n_("thermal")$. With damping $eta$:
   $ X = sqrt(eta) bb(1)_2, quad Y = (1 - eta)(n_("thermal") + 1/2) bb(1)_2 $

2. *Classical phase jitter (`classical_phase_jitter`):*
   Simulates a stochastic phase fluctuation in the small-angle approximation. This adds noise *exclusively* to the momentum quadrature $p$, while the position quadrature $q$ is preserved exactly:
   $ X = bb(1)_2, quad Y = mat(0, 0; 0, sigma_phi^2) $

These channel constructors produce mathematical channel objects; they become part of a circuit only when bound to a `Gate` and added to a `Circuit`. This separation keeps physical transformation semantics distinct from execution and serialization concerns.

---

=== Literature
The Gaussian-unitary and Gaussian-channel formulas in this chapter draw on the same background literature as Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005; Serafini 2023), which also cover the physical interpretation of symplectic transformations, loss and thermal channels, and the complete-positivity constraints on Gaussian channels.

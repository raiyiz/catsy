#import "@preview/physica:0.9.8": *

= Chapter 2: Gaussian Unitary Transformations & Channels

Gaussian operations are defined by the fact that they leave the Gaussian structure of the Wigner function invariant. Mathematically, unitary operators of this class correspond to affine symplectic transformations in phase space. Non-unitary processes (decoherence and noise) are modeled as CPTP maps (*Completely Positive Trace-Preserving Maps*) via Gaussian channels. This is the standard Gaussian-operation/channel framework of continuous-variable quantum information; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and #link("https://doi.org/10.1103/RevModPhys.77.513")[Braunstein and van Loock (2005)].

== Unitary gate transformations (`GaussianOperations`)
Every purely Gaussian unitary transformation $hat(U)$ induces a linear transformation in phase space, expressed via a symplectic matrix $S in S_p(2n, RR)$. For the displacement vector $d$ and the covariance matrix $V$:
$d -> S d$
$V -> S V S^T$

Preserving the canonical commutation relations strictly requires that $S$ preserve the symplectic form:
$S Omega S^T = Omega$

The toolkit implements three fundamental basis transformations in `GaussianOperations`:

1. *Squeezing operator ($hat(S)_k (r, theta)$):*
   The local squeeze operator on mode $k$ reduces the variance in one quadrature below the shot-noise limit, while amplifying the conjugate quadrature. Squeezing, passive rotations, and beam splitters are standard examples of Gaussian unitaries; their symplectic representation is developed systematically in the references above and in Serafini’s graduate-level treatment. The local symplectic matrix is:
   $ S_("local") = mat(cos(theta), -sin(theta); sin(theta), cos(theta)) mat(e^(-r), 0; 0, e^r) mat(cos(theta), sin(theta); -sin(theta), cos(theta)) $

2. *Phase rotation ($hat(R)_k (phi)$):*
   A passive, energy-preserving transformation that rotates phase space by angle $phi$:
   $ S_("local") = mat(cos(phi), -sin(phi); sin(phi), cos(phi)) $

3. *Beam splitter ($hat(B)_(k, m)(eta)$):*
   A lossless beam splitter couples modes $k$ and $m$ with power transmissivity $eta$. It generates entanglement between independent modes:
   $t = sqrt(eta), quad r_c = sqrt(1 - eta)$
   The transformation mixes the mode pair as:
   $mat(d_k ; d_m) -> mat(t bb(1)_2, r_c bb(1)_2; -r_c bb(1)_2, t bb(1)_2) mat(d_k ; d_m)$

== General Gaussian channels (`GaussianChannel`)
Noise and dissipation can no longer be represented by unitary $S$-matrices alone. A general Gaussian channel is mathematically described by two real matrices $X$ and $Y$:
$d -> X d + d_0$
$V -> X V X^T + Y$

For this transformation to be a physical CPTP map, the matrix $Y$ must satisfy the noise inequality:
$Y + i/2 Omega - i/2 X Omega X^T >= 0$

The toolkit implements these maps efficiently via a global coordinate embedding in the `GaussianChannel` class:

```python
@dataclass(frozen=True)
class GaussianChannel:
    """A general Gaussian channel d' = X@d + d0, V' = X@V@X.T + Y
    acting on a subset of modes."""
    target_modes: tuple[str, ...]
    X: np.ndarray = field(hash=False)
    Y: np.ndarray = field(hash=False)
    d0: np.ndarray = field(hash=False)

    def apply(self, state: GaussianState) -> GaussianState:
        global_dim = len(state.displacement)
        X_global = np.eye(global_dim)
        Y_global = np.zeros((global_dim, global_dim))
        d0_global = np.zeros(global_dim)

        # Embed the local matrices into the global mode indices
        for l_idx1, m1 in enumerate(self.target_modes):
            gi1 = state.get_mode_index(m1)
            d0_global[gi1 : gi1 + 2] = self.d0[l_idx1 * 2 : l_idx1 * 2 + 2]
            for l_idx2, m2 in enumerate(self.target_modes):
                gi2 = state.get_mode_index(m2)
                X_global[gi1 : gi1 + 2, gi2 : gi2 + 2] = self.X[
                    l_idx1 * 2 : l_idx1 * 2 + 2, l_idx2 * 2 : l_idx2 * 2 + 2
                ]
                Y_global[gi1 : gi1 + 2, gi2 : gi2 + 2] = self.Y[
                    l_idx1 * 2 : l_idx1 * 2 + 2, l_idx2 * 2 : l_idx2 * 2 + 2
                ]

        new_d = X_global @ state.displacement + d0_global
        new_V = X_global @ state.covariance @ X_global.T + Y_global
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)
```

== Standard optical noise channels (`LossChannels`)
The toolkit provides standard physical channels through the factory object `LossChannels`:

1. *Thermal loss channel (`thermal_loss`):*
   Models coupling to a thermal bath with mean photon number $n_("thermal")$. With damping $eta$:
   $ X = sqrt(eta) bb(1)_2, quad Y = (1 - eta)(n_("thermal") + 1/2) bb(1)_2 $

2. *Classical phase jitter (`classical_phase_jitter`):*
   Simulates a stochastic phase fluctuation in the small-angle approximation. This adds noise *exclusively* to the momentum quadrature $p$, while the position quadrature $q$ is preserved exactly:
   $ X = bb(1)_2, quad Y = mat(0, 0; 0, sigma_phi^2) $

---



== Literature
The Gaussian-unitary and Gaussian-channel formulas in this chapter draw on the same background literature as Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005; Serafini 2023), which also cover the physical interpretation of symplectic transformations, loss and thermal channels, and the complete-positivity constraints on Gaussian channels.

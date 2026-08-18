#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 4
// ==========================================
= Chapter 4: Quantum Measurement Processes & State Conditioning

Measurements on continuous variables induce a genuine, irreversible conditional state update of the overall quantum state. Since the toolkit operates in the Gaussian phase-space layer, the update can be expressed directly in terms of conditional Gaussian moments rather than explicit infinite-dimensional projection operators. The framework implements this using the covariance-matrix conditioning (Schur-complement) formulas standard in Gaussian quantum measurement theory. See #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and #link("https://doi.org/10.1103/RevModPhys.77.513")[Braunstein and van Loock (2005)].

== Homodyne measurement (`homodyne_measurement`)
Homodyne detection measures a freely chosen linear combination of the canonical quadrature operators $q$ and $p$ of a target mode. This process is intrinsically stochastic: the idealized measurement projects onto a quadrature eigenstate, while the remaining modes are *conditioned* on the measurement result. In a physical implementation, the conditional update is the relevant operational statement; the phase-space treatment avoids requiring an explicit infinite-energy eigenstate representation.

=== Mathematical transformation & mode rotation
To generalize the measurement to an arbitrary local-oscillator angle $phi$ (where $phi=0$ corresponds to the position quadrature $q$ and $phi=pi/2$ to the momentum quadrature $p$), the code first transforms the system via a global passive rotation matrix $R_("global")$ into the eigenbasis of the measurement apparatus. This reduces every homodyne detection mathematically to a pure measurement of the first quadrature ($q$) of the target mode.

The rotated covariance matrix $V_("rot")$ is partitioned into four structural submatrices:
$ V_("rot") = mat(V_(M M), V_(M R); V_(R M), V_(R R)) $

- $V_(M M)$ (scalar): the inherent variance of the quadrature being measured.
- $V_(M R)$ and $V_(R M)^T$ (vectors): the cross-correlations between the measured mode and all remaining system modes. This is the mathematical lever by which entanglement steers the collapse of the rest of the system.
- $V_(R R)$ (matrix): the isolated covariance matrix of the unmeasured submodes.

=== Stochastic sampling & Schur conditioning
If no explicit measurement value (`outcome`) is given, the toolkit computes the physically correct stochastic measurement outcome. The probability distribution of the outcome $x_m$ is a Gaussian centered on the rotated mean of the quadrature, with a width given by the quantum variance:
$x_m "sim" N(d_M, sqrt(V_(M M)))$

The conditional state of the remaining modes is then computed via the *Schur complement*, which is the classical-looking matrix form taken by Gaussian quantum conditioning. The evolution of the displacement vector $d_("cond")$ and the covariance matrix $V_("cond")$ is implemented in the code exactly as follows:

```python
@staticmethod
def homodyne_measurement(
    state: GaussianState,
    measured_mode: str,
    phi: float,
    outcome: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, GaussianState]:
    n_modes = len(state.modes)
    idx_m = state.get_mode_index(measured_mode)

    # 1. Build the global rotation operator for the LO angle phi
    R_local = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    R_global = np.eye(2 * n_modes)
    R_global[idx_m : idx_m + 2, idx_m : idx_m + 2] = R_local

    # Rotate the moments into the measurement basis
    d_rot = R_global @ state.displacement
    V_rot = R_global @ state.covariance @ R_global.T

    # Extract indices of all unmeasured quadratures
    idx_x = idx_m
    remaining_indices = [
        i for i in range(2 * n_modes) if i != idx_x and i != idx_m + 1
    ]

    # 2. Block partitioning for the Schur complement
    V_MM = V_rot[idx_x, idx_x]
    V_MR = V_rot[idx_x, remaining_indices]
    V_RM = V_rot[remaining_indices, idx_x]
    V_RR = V_rot[np.ix_(remaining_indices, remaining_indices)]

    # 3. Deterministic or stochastic sampling of the outcome
    if outcome is None:
        rng = rng if rng is not None else np.random.default_rng()
        measured_value = rng.normal(loc=d_rot[idx_x], scale=np.sqrt(V_MM))
    else:
        measured_value = outcome

    # 4. Compute the conditioned state (Wigner collapse)
    # The displacement shifts proportionally to the deviation from the expectation value
    d_cond = d_rot[remaining_indices] + V_RM * (1.0 / V_MM) * (
        measured_value - d_rot[idx_x]
    )
    # The new covariance matrix shrinks; uncertainty is reduced by information extraction
    V_cond = V_RR - np.outer(V_RM, V_MR) / V_MM

    remaining_modes = tuple(m for m in state.modes if m != measured_mode)
    return float(measured_value), GaussianState(remaining_modes, d_cond, V_cond)
```

=== Why the code does this (physical causality)
- *`V_RM * (1.0 / V_MM) * (measured_value - d_rot[idx_x])`*: if the measured mode was entangled with the rest of the system (e.g. an EPR pair), its state correlates with the other modes. If the measurement outcome `measured_value` deviates from its quantum-mechanical mean, this correlation forces the remaining system into a macroscopic shift in phase space.
- *`- np.outer(V_RM, V_MR) / V_MM`*: every homodyne measurement extracts information from the overall system. Since the cross-correlations $V_("RM")$ encode the amount of quantum knowledge about the subsystem, the Schur complement subtracts exactly this uncertainty. The remaining system shrinks in phase space along the entangled axes.

== Heterodyne measurement (`heterodyne_measurement`)
Heterodyne (or dual-homodyne) detection measures both conjugate quadratures ($q$ and $p$) of a mode simultaneously. Since $[q, p] = i != 0$, the Heisenberg uncertainty principle forbids an exact simultaneous measurement without injecting additional noise.

=== The mathematical vacuum-port model
Physically, heterodyne measurement corresponds to splitting the target mode on a 50:50 beam splitter whose second input is populated with an uncorrelated vacuum state. The two outputs are then each homodyne-detected (one measures $q$, the other $p$).

This intrinsic quantum noise is simulated elegantly and efficiently in the code, without needing to explicitly construct the beam splitter in phase space: the vacuum noise is added directly as an additive term to the measurement block:
$ V_("eff") = V_(M M) + 1/2 bb(1)_2 $

=== Implementation & noise injection
Because of this noise injection, the measured mode does not collapse onto an infinitely squeezed eigenstate, but onto a coherent state (projection onto coherent states / *Husimi Q representation*).

```python
@staticmethod
def heterodyne_measurement(
    state: GaussianState,
    measured_mode: str,
    outcome: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, GaussianState]:
    idx_m = state.get_mode_index(measured_mode)
    dim = len(state.displacement)
    remaining_indices = [i for i in range(dim) if i < idx_m or i > idx_m + 1]

    # Partition the pure system covariance (2x2 block for the target mode)
    V_MM = state.covariance[idx_m : idx_m + 2, idx_m : idx_m + 2]
    V_MR = state.covariance[idx_m : idx_m + 2, remaining_indices]
    V_RM = V_MR.T
    V_RR = state.covariance[np.ix_(remaining_indices, remaining_indices)]
    
    # Inject the minimal Heisenberg vacuum noise (0.5 * I_2)
    V_eff = V_MM + 0.5 * np.eye(2)
    V_eff_inv = np.linalg.inv(V_eff)

    # Multivariate sampling over the noisy distribution
    if outcome is None:
        rng = rng if rng is not None else np.random.default_rng()
        measured_vector = rng.multivariate_normal(
            mean=state.displacement[idx_m : idx_m + 2], cov=V_eff
        )
    else:
        measured_vector = np.asarray(outcome, dtype=float)

    # Matrix conditioning via the noisy Schur complement
    d_cond = state.displacement[remaining_indices] + V_RM @ V_eff_inv @ (
        measured_vector - state.displacement[idx_m : idx_m + 2]
    )
    V_cond = V_RR - V_RM @ V_eff_inv @ V_MR

    remaining_modes = tuple(m for m in state.modes if m != measured_mode)
    return measured_vector, GaussianState(remaining_modes, d_cond, V_cond)
```

=== Why the code does this (physical causality)
- *`V_eff = V_MM + 0.5 * np.eye(2)`*: the added `0.5 * np.eye(2)` represents exactly the fluctuation quantum of the unused beam-splitter input. Without this term, the resulting matrix inversion would be singular or physically underspecified for ideally squeezed states, leading to violations of the Robertson-Schrödinger uncertainty relation in the remaining state.
- *`V_cond = V_RR - V_RM @ V_eff_inv @ V_MR`*: because the measurement noise means less information can be extracted about the system than with a homodyne measurement, the modified `V_eff_inv` ensures the variances of the remaining system $V_("cond")$ shrink less strongly. The eigenvalues of the resulting covariance matrix remain guaranteed to stay above the vacuum limit ($>= 0.5$).

---


== Scientific literature
For homodyne and heterodyne detection, Gaussian conditioning, and continuous-variable measurements, see:

- #link("https://doi.org/10.1103/RevModPhys.84.621")[C. Weedbrook et al., “Gaussian quantum information,” *Reviews of Modern Physics* 84, 621–669 (2012).]
- #link("https://doi.org/10.1103/RevModPhys.77.513")[S. L. Braunstein and P. van Loock, “Quantum information with continuous variables,” *Reviews of Modern Physics* 77, 513–577 (2005).]
- #link("https://www.routledge.com/Quantum-Continuous-Variables-A-Primer-of-Theoretical-Methods/Serafini/p/book/9781032157238")[A. Serafini, *Quantum Continuous Variables: A Primer of Theoretical Methods*, 2nd ed. (CRC Press, 2024).]
- #link("https://doi.org/10.1002/3527602976.ch3")[W. P. Schleich, *Quantum Optics in Phase Space*, especially the chapters on Wigner functions and quantum-state reconstruction.]

These references also provide useful background for the distinction between ideal homodyne projection and the noisy coherent-state POVM associated with heterodyne detection.

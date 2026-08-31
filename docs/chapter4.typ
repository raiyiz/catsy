#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 4
// ==========================================
= Chapter 4: Quantum Measurement Processes & State Conditioning

Measurements on continuous variables induce an irreversible conditional state update of the overall quantum state. Since the toolkit operates in the Gaussian phase-space layer, the update can be expressed directly in terms of conditional Gaussian moments rather than explicit infinite-dimensional projection operators. The framework implements this using the covariance-matrix conditioning (Schur-complement) formulas standard in Gaussian quantum measurement theory. See #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and #link("https://doi.org/10.1103/RevModPhys.77.513")[Braunstein and van Loock (2005)].

In the terminology established earlier, these measurement routines are operations on a `GaussianState`: they transform the state and, for conditional measurements, return the measurement outcome together with the conditioned state. They are not `Gate` objects, because a measurement consumes information from the state rather than representing a reusable unitary gate in a `Circuit`.

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
    if not np.isfinite(phi):
        raise ValueError(f"phi must be finite, got {phi!r}.")
    if outcome is not None and (
        not isinstance(outcome, int | float) or not np.isfinite(outcome)
    ):
        raise ValueError("homodyne outcome must be a finite scalar.")

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
    if not np.isfinite(V_MM) or V_MM <= TOL_PHYSICALITY:
        raise ValueError(
            f"homodyne measurement variance must be finite and positive; got {V_MM:.3e}."
        )
    V_MR = V_rot[idx_x, remaining_indices]
    V_RM = V_rot[remaining_indices, idx_x]
    V_RR = V_rot[np.ix_(remaining_indices, remaining_indices)]

    d_M = d_rot[idx_x]
    d_R = d_rot[remaining_indices]

    # 3. Deterministic or stochastic sampling of the outcome
    if outcome is None:
        rng = rng if rng is not None else np.random.default_rng()
        measured_value = rng.normal(loc=d_M, scale=np.sqrt(V_MM))
    else:
        measured_value = outcome

    # 4. Compute the conditioned state (Wigner collapse)
    # The displacement shifts proportionally to the deviation from the expectation value
    gain = V_RM / V_MM
    d_cond = d_R + gain * (measured_value - d_M)
    # The new covariance matrix shrinks; uncertainty is reduced by information extraction
    V_cond = V_RR - np.outer(V_RM, V_MR) / V_MM
    V_cond = 0.5 * (V_cond + V_cond.T)

    remaining_modes = tuple(m for m in state.modes if m != measured_mode)
    return measured_value, GaussianState(remaining_modes, d_cond, V_cond)
```

=== Why the code does this (physical causality)
- *validating `phi`, `outcome`, and `V_MM` up front*: a non-finite local-oscillator angle, a non-finite forced outcome, or a numerically zero/negative measured-quadrature variance would otherwise propagate silently into a division by (near) zero a few lines later; the code fails fast with a specific `ValueError` instead.
- *`gain * (measured_value - d_M)`*, where `gain = V_RM / V_MM`: if the measured mode was entangled with the rest of the system (e.g. an EPR pair), its state correlates with the other modes. If the measurement outcome `measured_value` deviates from its quantum-mechanical mean, this correlation forces the remaining system into a macroscopic shift in phase space.
- *`- np.outer(V_RM, V_MR) / V_MM`*: every homodyne measurement extracts information from the overall system. Since the cross-correlations $V_("RM")$ encode the amount of quantum knowledge about the subsystem, the Schur complement subtracts exactly this uncertainty. The remaining system shrinks in phase space along the entangled axes.
- *`V_cond = 0.5 * (V_cond + V_cond.T)`*: the Schur-complement formula is exactly symmetric in exact arithmetic, but floating-point round-off can leave a tiny antisymmetric residual; explicitly symmetrizing keeps `V_cond` a valid covariance matrix for the `GaussianState` constructor's own symmetry check.

== Heterodyne measurement (`heterodyne_measurement`)
Heterodyne (or dual-homodyne) detection measures both conjugate quadratures ($q$ and $p$) of a mode simultaneously. Since $[q, p] = i != 0$, the Heisenberg uncertainty principle forbids an exact simultaneous measurement without injecting additional noise.

=== The mathematical vacuum-port model
Physically, heterodyne measurement corresponds to splitting the target mode on a 50:50 beam splitter whose second input is populated with an uncorrelated vacuum state. The two outputs are then each homodyne-detected (one measures $q$, the other $p$).

This intrinsic quantum noise is simulated directly in the code, without needing to explicitly construct the beam splitter in phase space: the vacuum noise is added directly as an additive term to the measurement block:
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
    V_RR = state.covariance[np.ix_(remaining_indices, remaining_indices)]
    d_M = state.displacement[idx_m : idx_m + 2]
    d_R = state.displacement[remaining_indices]

    # Inject the minimal Heisenberg vacuum noise (0.5 * I_2), symmetrized
    # against floating-point round-off
    V_eff = 0.5 * (V_MM + 0.5 * np.eye(2)) + 0.5 * (V_MM + 0.5 * np.eye(2)).T
    if not np.all(np.isfinite(V_eff)):
        raise ValueError("heterodyne effective covariance must be finite.")
    try:
        np.linalg.cholesky(V_eff)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "heterodyne effective covariance must be positive definite."
        ) from exc

    # Multivariate sampling over the noisy distribution
    if outcome is None:
        rng = rng if rng is not None else np.random.default_rng()
        measured_outcome = rng.multivariate_normal(mean=d_M, cov=V_eff)
    else:
        measured_outcome = np.asarray(outcome, dtype=float)
        if measured_outcome.shape != (2,):
            raise ValueError(
                f"heterodyne outcome must have shape (2,), got {measured_outcome.shape}."
            )
        if not np.all(np.isfinite(measured_outcome)):
            raise ValueError("heterodyne outcome must contain only finite values.")

    # Matrix conditioning via the noisy Schur complement (solve, not an
    # explicit inverse, for numerical stability)
    gain = np.linalg.solve(V_eff, V_MR).T
    innovation = measured_outcome - d_M
    d_cond = d_R + gain @ innovation
    V_cond = V_RR - gain @ V_MR
    V_cond = 0.5 * (V_cond + V_cond.T)

    remaining_modes = tuple(m for m in state.modes if m != measured_mode)
    return measured_outcome, GaussianState(remaining_modes, d_cond, V_cond)
```

=== Why the code does this (physical causality)
- *`V_eff = V_MM + 0.5 * np.eye(2)`* (symmetrized): the added `0.5 * np.eye(2)` represents exactly the fluctuation quantum of the unused beam-splitter input. The explicit finiteness and Cholesky positive-definiteness checks turn what would otherwise be a silent singular-matrix failure -- or a physically underspecified result for ideally squeezed states -- into an early, specific `ValueError`.
- *`gain = np.linalg.solve(V_eff, V_MR).T`*: solving the linear system for the gain avoids explicitly forming `V_eff`'s inverse, which is the numerically preferred way to apply $V_(M M)^(-1)$ without amplifying round-off error.
- *`V_cond = V_RR - gain @ V_MR`* (symmetrized): because the measurement noise means less information can be extracted about the system than with a homodyne measurement, the Schur complement here removes less uncertainty than the homodyne case. The eigenvalues of the resulting covariance matrix remain guaranteed to stay above the vacuum limit ($>= 0.5$); the final symmetrization guards against floating-point asymmetry the same way it does for homodyne conditioning above.

---


=== Literature
For homodyne and heterodyne detection and Gaussian conditioning, see the general background in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005; Serafini 2023), plus specifically:

- #link("https://doi.org/10.1002/3527602976.ch3")[W. P. Schleich, *Quantum Optics in Phase Space*, especially the chapters on Wigner functions and quantum-state reconstruction.]

This reference provides useful background for the distinction between ideal homodyne projection and the noisy coherent-state POVM associated with heterodyne detection.

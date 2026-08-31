#import "@preview/physica:0.9.8"

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
- $V_(M R)$ and $V_(R M)^T$ (vectors): the cross-correlations between the measured mode and all remaining system modes. This is the mathematical lever by which correlations steer the conditional update of the rest of the system.
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
    ...
```

=== Why the code does this (physical causality)
- *validating `phi`, `outcome`, and `V_MM` up front*: a non-finite local-oscillator angle, a non-finite forced outcome, or a numerically zero/negative measured-quadrature variance would otherwise propagate silently into a division by (near) zero a few lines later; the code fails fast with a specific `ValueError` instead.
- *`gain * (measured_value - d_M)`*, where `gain = V_RM / V_MM`: if the measured mode was correlated with the rest of the system (e.g. an EPR pair), its state correlates with the other modes. If the measurement outcome `measured_value` deviates from its quantum-mechanical mean, this correlation shifts the remaining system in phase space.
- *`- np.outer(V_RM, V_MR) / V_MM`*: the Schur complement subtracts the uncertainty explained by the measured quadrature and its correlations with the remaining subsystem.
- *`V_cond = 0.5 * (V_cond + V_cond.T)`*: the Schur-complement formula is exactly symmetric in exact arithmetic, but floating-point round-off can leave a tiny antisymmetric residual; explicitly symmetrizing keeps `V_cond` a valid covariance matrix for the `GaussianState` constructor's own symmetry check.

== Heterodyne measurement (`heterodyne_measurement`)
Heterodyne (or dual-homodyne) detection measures both conjugate quadratures ($q$ and $p$) of a mode simultaneously. Since $[q, p] = i != 0$, the Heisenberg uncertainty principle forbids an exact simultaneous measurement without injecting additional noise.

=== The mathematical vacuum-port model
Physically, heterodyne measurement corresponds to splitting the target mode on a 50:50 beam splitter whose second input is populated with an uncorrelated vacuum state. The two outputs are then each homodyne-detected (one measures $q$, the other $p$).

This intrinsic quantum noise is simulated directly in the code, without needing to explicitly construct the beam splitter in phase space: the vacuum noise is added directly as an additive term to the measurement block:
$ V_("eff") = V_(M M) + 1/2 bb(1)_2 $

=== Implementation & noise injection
Because of this noise injection, heterodyne detection is represented by a noisy coherent-state POVM rather than an infinitely sharp quadrature projection. The conditioned state of the *unmeasured* modes is obtained with the corresponding noisy Schur complement.

The implementation uses the effective covariance $V_("eff") = V_("MM") + I/2$ for both outcome sampling and conditioning. The solve-based gain computation avoids explicitly forming a matrix inverse.

=== Why the code does this (physical causality)
- *`V_eff = V_MM + 0.5 * np.eye(2)`*: the added `0.5 * I` represents the vacuum fluctuations entering the unused input port of the 50:50 heterodyne beam splitter in the package's covariance convention.
- *`gain = np.linalg.solve(V_eff, V_MR).T`*: solving the linear system for the gain avoids explicitly forming `V_eff`'s inverse and is numerically preferable.
- *`V_cond = V_RR - gain @ V_MR`*: the added measurement noise makes heterodyne conditioning less informative than the corresponding noiseless joint-quadrature conditioning. It does *not* imply that every eigenvalue of the conditioned remote covariance is bounded below by `0.5`; the conditional state may itself be squeezed. The universal statement is about the measurement noise model, not a vacuum lower bound on the conditioned subsystem.

---

=== Literature
For homodyne and heterodyne detection and Gaussian conditioning, see the general background in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005; Serafini 2023), plus specifically:

- #link("https://doi.org/10.1002/3527602976.ch3")[W. P. Schleich, *Quantum Optics in Phase Space*, especially the chapters on Wigner functions and quantum-state reconstruction.]

This reference provides useful background for the distinction between ideal homodyne projection and the noisy coherent-state POVM associated with heterodyne detection.

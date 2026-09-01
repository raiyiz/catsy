#import "@preview/physica:0.9.8": *
#import "links.typ": src-link

// ==========================================
// CHAPTER 7
// ==========================================
= Chapter 7: Non-Gaussian Gates & Physical Simulations

Not every interesting operation in CV quantum optics stays within the Gaussian class. Photon subtraction/addition, Kerr nonlinearities, and photon-number-resolving observables generate or require Fock-space structure. The Fock-space operations live in `src/catsy/fock/__init__.py`, while `src/catsy/optics.py` implements concrete, time-resolved hardware models (a driven Kerr cavity and a Mach-Zehnder interferometer). Reusable Gaussian gate layouts (Chapter 8) live on `Circuit` itself.

The representation boundary is explicit: `GaussianState.to_fock()` (Chapter 5) converts a compact Gaussian state into the truncated Fock representation needed here. `FockState` is the domain-level representation of that Fock-space state, while QuTiP supplies the underlying density-matrix/operator machinery. A Fock representation is not synonymous with a non-Gaussian state: Gaussian states can also be represented in Fock space after conversion. What distinguishes this chapter is that these operations require Hilbert-space structure that cannot be retained in the compact $(d, V)$ Gaussian representation.

== Primitive photon operations

The Fock module operates directly on QuTiP density matrices rather than on `GaussianState`. The public API is functional: photon operations are exposed as module-level functions, while `FockGates` remains as a backwards-compatible namespace for existing callers. At the domain level, these functions are the Fock-space operations applied to a `FockState` representation.

=== Subtraction and addition

Applying the annihilation or creation operator to a density matrix is in general not a trace-preserving process — it models a *heralded*, success-conditioned operation. The ideal operations are exposed as `photon_subtraction` and `photon_addition`:

```python
def photon_subtraction(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply textbook photon subtraction ``rho -> a rho a†``."""


def photon_addition(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
) -> qt.Qobj:
    """Apply textbook photon addition ``rho -> a† rho a``."""
```

Photon subtraction is commonly implemented conceptually by a weakly reflecting beam splitter followed by conditional detection in the reflected arm; see #link("https://doi.org/10.1103/PhysRevA.61.032302")[Opatrný, Kurizki, and Welsch (2000)]. Both ideal operations are normalized after application. The normalization denominator is the physical heralding success probability; if it is numerically zero, the operation raises `ValueError` rather than producing an invalid state.

All Fock operations accept `mode_idx` for multimode states and `N_cutoff` as an optional consistency check. The state validator requires a QuTiP operator and equal Fock dimensions across all modes. The selected mode is acted on locally, with the remaining modes preserved for operations and traced out for single-mode visualization.

=== Realistic heralded operations

The module also provides `realistic_photon_subtraction` and `realistic_photon_addition`. These model a weak coupling to an ancilla mode, followed by an imperfect click detector, rather than replacing the laboratory process by the ideal $hat(a)$ or $hat(a)^dagger$ map. The detector efficiency is therefore an explicit parameter. In the weak-coupling, high-efficiency limit these operations converge toward their ideal counterparts.

```python
def realistic_photon_subtraction(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
    tap_reflectivity: float = 0.05,
    detector_efficiency: float = 0.6,
    ancilla_cutoff: int = 6,
) -> qt.Qobj:
    """Heralded photon subtraction via a beamsplitter tap + click detector."""


def realistic_photon_addition(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
    coupling_strength: float = 0.05,
    detector_efficiency: float = 0.6,
    ancilla_cutoff: int = 6,
) -> qt.Qobj:
    """Heralded photon addition via parametric coupling + click detector."""
```

=== Photon-number observables

`mean_photon_number` returns the expectation value of the selected mode's number operator. `photon_number_measurement` performs an ideal photon-number-resolving measurement and returns the selected integer outcome together with the collapsed state. For a multimode input, the measured mode is removed from the returned state by partial trace.

```python
def mean_photon_number(
    rho: qt.Qobj, mode_idx: int = 0, N_cutoff: int | None = None,
) -> float:
    """Return <n> = tr(rho * a-dagger a) for the selected mode."""


def photon_number_measurement(
    rho: qt.Qobj,
    mode_idx: int = 0,
    N_cutoff: int | None = None,
    outcome: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, qt.Qobj]:
    """Ideal photon-number-resolving detection on mode_idx."""
```

For backward compatibility, existing code can continue to call these operations through `FockGates`, for example `FockGates.photon_subtraction(rho)`. New code should prefer the module-level functions directly.

== Driven, dissipative Kerr cavity (`KerrCavity`)

`KerrCavity` (in #src-link("src/catsy/optics.py", line: 612, label: [`optics.py`])) simulates a single optical cavity with Kerr nonlinearity $K$, photon loss rate $kappa$, and a time-dependent classical drive — a standard nonlinear model for generating and evolving non-classical states beyond the Gaussian class. Kerr evolution is closely associated with nonclassical collapse/revival dynamics and multi-component cat-like states; see #link("https://doi.org/10.1038/nature11902")[Kirchmair et al. (2013)]. The Hamiltonian is composed of the Kerr term and a Gaussian-shaped pulsed drive:
$ hat(H)(t) = K hat(a)^(dagger 2) hat(a)^2 + Omega(t)(hat(a) + hat(a)^dagger), quad Omega(t) = A exp(-(t - t_0)^2 / (2 sigma^2)) $

```python
def __init__(self, K: float, kappa: float, N_cutoff: int): ...
```

`K` is the Kerr strength, `kappa` the cavity photon-loss rate, and `N_cutoff` the Fock-space Hilbert-space dimension used throughout the simulation.

Dissipation is modeled via a single Lindblad collapse operator $sqrt(kappa) hat(a)$, and the full master equation is integrated in time with `qutip.mesolve`:

```python
def run(self, rho_init, tlist, amp, t0, sigma):
    a = qt.destroy(self.N_cutoff)
    H_kerr = self.K * a.dag() * a.dag() * a * a

    def pulse_shape(t, amp, t0, sigma):
        return amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    H_total = [H_kerr, [a + a.dag(), pulse_shape]]
    c_ops = [np.sqrt(self.kappa) * a] if self.kappa > 0 else []
    args = {"amp": float(amp), "t0": float(t0), "sigma": float(sigma)}

    result = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args=args)
    return result.states
```

The time-dependent part `[a + a.dag(), pulse_shape]` follows QuTiP's standard convention for time-dependent Hamiltonians: a list of (time-independent operator, time function) pairs, which `mesolve` assembles into $hat(H)(t) = hat(H)_"kerr" + Omega(t)(hat(a) + hat(a)^dagger)$. For $kappa = 0$, the collapse-operator list is left empty, so `mesolve` automatically falls back to a purely unitary Schrödinger integration.

== Mach-Zehnder interferometer with a lossy arm (`MachZehnderInterferometer`)

`MachZehnderInterferometer` (in #src-link("src/catsy/optics.py", line: 680, label: [`optics.py`])) models a two-mode interferometer in which an input state (e.g. a Schrödinger-cat state plus vacuum in the second port) passes through the sequence
$ "50:50 BS" arrow.r "lossy arm (fixed time)" arrow.r "phase shift" theta arrow.r "50:50 BS" $
and is then read out in a photon-number- and parity-resolved way.

```python
def __init__(self, kappa: float, N_cutoff: int, *, loss_time: float = 1.0): ...
```

`kappa` is the photon-loss rate in the lossy arm, `N_cutoff` the per-mode Fock-space dimension, and `loss_time` the fixed physical exposure time of the lossy arm (applied before the scanned phase, so its strength stays independent of $theta$). The first beam splitter is constructed as an exact QuTiP unitary operator:
$ hat(U)_"BS" = exp(i pi/4 (hat(a)_1^dagger hat(a)_2 + hat(a)_1 hat(a)_2^dagger)) $

```python
a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
a2 = qt.tensor(qt.qeye(N), qt.destroy(N))
n1_op = a1.dag() * a1
n2_op = a2.dag() * a2
parity1_op = (1j * np.pi * n1_op).expm()

U_BS = ((1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())).expm()
```

The loss is deliberately applied *before* the scanned phase shift, propagated with `mesolve` over a fixed physical exposure time `loss_time` — this keeps the loss strength independent of the phase $theta$ currently under consideration, which is essential for a clean phase scan:

```python
c_ops = [np.sqrt(self.kappa) * a1] if self.kappa > 0 and self.loss_time > 0 else []
if c_ops:
    loss_sim = qt.mesolve(0 * n1_op, psi_after_BS1, [0.0, self.loss_time], c_ops=c_ops)
    rho_after_loss = loss_sim.states[-1]
```

Afterwards, for each value of the supplied phase list `theta_list`, the phase operator $hat(U)_theta = exp(i theta hat(n)_1)$ is applied to the (already lossy) state, followed by the second beam splitter; the outputs read out are the mean photon number of each port ($chevron.l hat(n)_1 chevron.r$, $chevron.l hat(n)_2 chevron.r$) and the parity expectation $chevron.l exp(i pi hat(n)_1) chevron.r$ of the first output — the latter being a parity-sensitive interference observable that is particularly useful for resolving nonclassical phase-space structure and cat-like states. Its interpretation should be understood as a metrological/diagnostic choice rather than as a universal definition of a cat state.

---


=== Literature
The non-Gaussian operations and dynamical models in this chapter are connected to the following primary literature; Weedbrook et al. (2012, Chapter 1) provides the Gaussian/non-Gaussian boundary that motivates the chapter:

- #link("https://doi.org/10.1103/PhysRevA.61.032302")[T. Opatrný, G. Kurizki, and D.-G. Welsch, “Improvement on teleportation of continuous variables by photon subtraction via conditional measurement,” *Physical Review A* 61, 032302 (2000).]
- #link("https://doi.org/10.1103/PhysRevA.72.033822")[J. Fiurášek, R. García-Patrón, and N. J. Cerf, “Conditional generation of arbitrary single-mode quantum states of light by repeated photon subtractions,” *Physical Review A* 72, 033822 (2005).]
- #link("https://doi.org/10.1038/nature11902")[G. Kirchmair et al., “Observation of quantum state collapse and revival due to the single-photon Kerr effect,” *Nature* 495, 205–209 (2013).]

The first two references are particularly relevant to the physical interpretation of the `a rho a^dagger` and `a^dagger rho a` maps as conditional operations. The Kerr reference provides an experimental benchmark for the collapse/revival physics represented by `KerrCavity`.

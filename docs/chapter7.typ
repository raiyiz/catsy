#import "@preview/physica:0.9.8": *
#import "links.typ": src-link

// ==========================================
// CHAPTER 7
// ==========================================
= Chapter 7: Non-Gaussian Operations & Physical Simulations

Not every interesting operation in CV quantum optics stays within the Gaussian class. Photon subtraction/addition, Kerr nonlinearities, and photon-number-resolving observables generate or require Fock-space structure. These operations consistently live outside `gaussian.py`: #src-link("src/catsy/fock.py") provides primitive photon operations on already-existing QuTiP states, while #src-link("src/catsy/optics.py") implements concrete, time-resolved hardware models (a driven Kerr cavity, a Mach-Zehnder interferometer) alongside the Gaussian-optics `OpticalSetup` layouts -- both describe specific pieces of optical hardware rather than generic phase-space transformations, which is why they share a module.

== Primitive photon operations (`FockOperations`)

`FockOperations` deliberately operates *not* on `GaussianState`, but directly on QuTiP density matrices — the conversion is expected to be performed explicitly via `GaussianState.to_qutip()` (Chapter 5). This separation keeps the Fock layer lean and avoids a second, competing convenience layer for conversion.

=== Subtraction and addition

Applying the annihilation or creation operator to a density matrix is in general not a trace-preserving process — it models a *heralded*, success-conditioned operation. Photon subtraction is commonly implemented conceptually by a weakly reflecting beam splitter followed by conditional detection in the reflected arm; see #link("https://doi.org/10.1103/PhysRevA.61.032302")[Opatrný, Kurizki, and Welsch (2000)]. The state must therefore be renormalized after application:
$ rho -->^"subtraction" (hat(a) rho hat(a)^dagger) / "tr"(hat(a) rho hat(a)^dagger), quad rho -->^"addition" (hat(a)^dagger rho hat(a)) / "tr"(hat(a)^dagger rho hat(a)) $

```python
@staticmethod
def _apply_and_renormalize(rho, op, label: str):
    rho_new = op * rho * op.dag()
    trace_val = rho_new.tr()
    if abs(trace_val) < TOL_PHYSICALITY:
        raise ValueError(
            f"{label}: heralding success probability is numerically zero."
        )
    return rho_new / trace_val

@staticmethod
def photon_subtraction(rho, mode_idx: int = 0, N_cutoff: int = 20):
    n_modes = FockOperations._validate_state(rho, N_cutoff, mode_idx)
    a_op = FockOperations._mode_operator(
        qt.destroy(N_cutoff), n_modes, mode_idx, N_cutoff
    )
    return FockOperations._apply_and_renormalize(rho, a_op, "photon_subtraction")
```

The denominator $"tr"(hat(a) rho hat(a)^dagger)$ is simultaneously the physical *heralding success probability*: if it drops below the numerical tolerance `TOL_PHYSICALITY`, the renormalization would be singular (e.g. when attempting to subtract a photon from the vacuum), and the method aborts in a controlled way with a `ValueError` rather than silently producing `NaN` values. `_mode_operator` embeds the local $hat(a)$ or $hat(a)^dagger$ operator via `qt.tensor` at the correct mode position, provided `rho` has more than one mode; `_validate_state` checks beforehand that `rho` is indeed an operator whose Fock dimensions match `N_cutoff`.

== Driven, dissipative Kerr cavity (`KerrCavity`)

`KerrCavity` (in #src-link("src/catsy/optics.py", line: 330, label: [`optics.py`])) simulates a single optical cavity with Kerr nonlinearity $K$, photon loss rate $kappa$, and a time-dependent classical drive — a standard nonlinear model for generating and evolving non-classical states beyond the Gaussian class. Kerr evolution is closely associated with nonclassical collapse/revival dynamics and multi-component cat-like states; see #link("https://doi.org/10.1038/nature11902")[Kirchmair et al. (2013)]. The Hamiltonian is composed of the Kerr term and a Gaussian-shaped pulsed drive:
$ hat(H)(t) = K hat(a)^(dagger 2) hat(a)^2 + Omega(t)(hat(a) + hat(a)^dagger), quad Omega(t) = A exp(-(t - t_0)^2 / (2 sigma^2)) $

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

`MachZehnderInterferometer` (in #src-link("src/catsy/optics.py", line: 398, label: [`optics.py`])) models a two-mode interferometer in which an input state (e.g. a Schrödinger-cat state plus vacuum in the second port) passes through the sequence
$ "50:50 BS" arrow.r "lossy arm (fixed time)" arrow.r "phase shift" theta arrow.r "50:50 BS" $
and is then read out in a photon-number- and parity-resolved way. The first beam splitter is constructed as an exact QuTiP unitary operator:
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

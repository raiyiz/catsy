#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 5
// ==========================================
= Chapter 5: State Preparation & the Gaussian-Fock Bridge

While Chapters 2–4 deal with the transformation of *existing* states, this chapter is devoted to their *creation*: the constructors on `GaussianState` provide standard resources such as vacuum, coherent states, and two-mode squeezed vacuum (TMSV). It also introduces the explicit representation boundary `GaussianState.to_fock()`, which bridges the compact Gaussian phase-space representation into the truncated Fock-space representation used for non-Gaussian work.

== Constructors for standard resources (`GaussianState`)

Each factory method constructs a state *declaratively* from the vacuum, by composing the unitary transformations already introduced in Chapter 2. This keeps the physics of every resource traceable to a single place.

=== Vacuum and coherent states

The $n$-mode vacuum state trivially has no displacement and the isotropic shot-noise covariance:
$ d = 0, quad V = 1/2 bb(1)_(2n) $

A coherent state $ket(alpha)$ is a vacuum displaced by $alpha$. The `GaussianState.coherent` constructor creates this state directly:

```python
@classmethod
def coherent(
    cls, modes: tuple[str, ...], alphas: complex | Sequence[complex]
) -> GaussianState:
    """Return a multi-mode coherent state.

    A single scalar amplitude is broadcast to every mode; otherwise pass
    one amplitude per mode.
    """
    alpha_list: list[complex]
    if isinstance(alphas, int | float | complex):
        alpha_list = [complex(alphas)] * len(modes)
    else:
        alpha_list = list(alphas)
    if len(alpha_list) != len(modes):
        raise ValueError(
            f"Got {len(alpha_list)} alpha(s) for {len(modes)} mode(s); "
            "pass one alpha per mode (or a single scalar to broadcast)."
        )

    state = cls.vacuum(modes)
    for mode, alpha in zip(modes, alpha_list, strict=True):
        state = state.displace(mode, alpha)
    return state
```

A single scalar `alphas` is broadcast across all modes simultaneously — useful, e.g., for creating an $n$-mode reference beam with identical amplitude without manually repeating the list.

=== Two-mode squeezed vacuum (`GaussianState.tmsv`)

The standard recipe for non-classical continuous-variable entanglement first creates two orthogonally squeezed vacua and then mixes them on a 50:50 beam splitter. `GaussianState.tmsv` packages this standard two-mode-squeezing/EPR resource construction in CV quantum information; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)] and #link("https://www.routledge.com/Quantum-Continuous-Variables-A-Primer-of-Theoretical-Methods/Serafini/p/book/9781032157238")[Serafini (2023)].

```python
@classmethod
def tmsv(cls, mode_a: str, mode_b: str, r: float) -> GaussianState:
    return (
        cls.vacuum((mode_a, mode_b))
        .squeeze(mode_a, r=r, theta=0.0)
        .squeeze(mode_b, r=r, theta=np.pi / 2)
        .beam_splitter(mode_a, mode_b, eta=0.5)
    )
```

Mode `mode_a` is squeezed in $q$ ($theta=0$), mode `mode_b` in $p$ ($theta=pi/2$); the subsequent beam splitter fully entangles both modes. For any $r > 0$, the result satisfies:
$ "Var"(q_a - q_b) = "Var"(p_a + p_b) = e^(-2r) < 1 $

Both combined quadratures thus lie below the shot-noise limit of $1$ — a necessary signature of EPR correlation, quantitatively verified in Chapter 6 via the Duan-Simon witness. It is important to distinguish this from classically correlated states (e.g. from `LossChannels.correlated_thermal_noise`, Chapter 2): these can look similar in a scatter plot, but never violate the Duan-Simon bound.

== The Gaussian-Fock bridge: `GaussianState.to_fock()`

The phase-space layer stores a state compactly as $(d, V) in RR^(2n) times RR^(2n times 2n)$. For non-Gaussian operations (photon subtraction, Kerr nonlinearity, …), however, one needs the full density matrix $rho$ in a truncated Fock space. `GaussianState.to_fock()` is the explicit representation boundary: it converts the compact Gaussian state into a finite Fock-space density matrix. QuTiP supplies the underlying operator representation, but the conceptual boundary in catsy is between `GaussianState` and the Fock representation, not between `GaussianState` and QuTiP. The legacy `to_qutip()` name should be treated as a deprecated compatibility alias rather than the canonical API.

```python
def to_fock(self, N_cutoff: int = 15) -> FockState: ...
```

The conversion proceeds in three steps, each with its own numerical-stability considerations below.

=== Step 1 — Williamson decomposition

Every physical covariance matrix can be symplectically diagonalized:
$ V = S D S^T, quad D = "diag"(nu_1, nu_1, dots, nu_n, nu_n), quad S in "Sp"(2n, RR) $
The *symplectic eigenvalues* $nu_k >= 1/2$ are the invariants of the state under Gaussian unitaries: $nu_k = 1/2$ for all $k$ if and only if the state is pure. Williamson normal form is a key theorem for Gaussian states and is treated in detail by Serafini and Weedbrook et al. The diagonal state $rho_D$ is thus a pure tensor product of thermal states with mean photon number $overline(n)_k = nu_k - 1/2$:

```python
symplectic_values, S, D = _williamson_decomposition(self.covariance)
...
rho_list = [
    qt.thermal_dm(N_cutoff, max(float(nu) - 0.5, 0.0))
    for nu in symplectic_values
]
rho = qt.tensor(*rho_list)
```

The decomposition itself (`_williamson_decomposition` in `core.py`) avoids a naive matrix square root of $V D^(-1)$ and instead constructs $S$ via a real Schur decomposition of $sqrt(V) Omega sqrt(V)$ — numerically more stable, with explicit residual checks against $S Omega S^T = Omega$ and $S D S^T = V$.

=== Step 2 — Polar decomposition of the symplectic transformation

The remaining task is to transform the pure product state $rho_D$ via the symplectic matrix $S$ into the target state. Rather than forming $log(S)$ directly — which does not yield a well-defined quadratic generator for a general symplectic matrix — the code uses the *polar decomposition*
$ S = P dot O $
where $O$ is orthogonal-symplectic (a *passive*, energy-preserving transformation such as phase rotations and beam splitters) and $P$ is positive-definite symplectic (the pure *active* squeezing part). Both factors have well-defined Hermitian generators:

```python
P = scipy.linalg.sqrtm(S @ S.T).real
P_inv = scipy.linalg.inv(P)
O = P_inv @ S
```

The passive part is realized as a photon-number-conserving Hamiltonian $H_"passive" = sum_(i j) h_(i j) hat(a)_i^dagger hat(a)_j$ (see `_qutip_passive_unitary`), whose generating matrix $h$ follows from the complex logarithm of a unitary matrix constructed from $O$. The positive (squeezing) part is implemented as a quadratic quadrature operator
$ H_"positive" = 1/2 sum_(i j) G_(i j) hat(r)_i hat(r)_j, quad G = -Omega log(P) $
and likewise exponentiated. Both unitaries are applied in sequence, so that the total transformation $P dot O = S$ is reproduced exactly — both intermediate steps are checked in the code against the residuals $O^T O = bb(1)$, $O Omega O^T = Omega$, and $P Omega P^T = Omega$.

=== Step 3 — Displacement

Finally, the displacement operator is applied per mode. QuTiP's convention $alpha = (d_x + i d_p) / sqrt(2)$ exactly matches the convention of the phase-space layer, so no further conversion is required:

```python
alpha = (dx + 1j * dp) / np.sqrt(2.0)
op_list = [qt.qeye(N_cutoff) for _ in range(n_modes)]
op_list[i] = qt.displace(N_cutoff, alpha)
D_op = qt.tensor(*op_list)
rho = D_op * rho * D_op.dag()
```

=== Limits of the conversion

The Williamson decomposition is mathematically exact; the implementation verifies this to floating-point tolerance. The returned Fock-space density matrix, however, lives in a finite Fock cutoff `N_cutoff`, so the transition from phase space to Hilbert space still introduces a truncation error for strongly squeezed or highly populated states. In practice, `N_cutoff` should be chosen so that the occupation probability at the upper edge of the Fock space is negligible.

== Persistence and mode ordering

Beyond the Fock bridge, `GaussianState` offers a lightweight JSON serialization (`to_dict`/`from_dict`, `save`/`load`) as well as `reorder_modes`, which permutes the displacement vector and covariance matrix together along a new mode order:

```python
def reorder_modes(self, modes: tuple[str, ...] | list[str]) -> GaussianState:
    requested = tuple(modes)
    if len(requested) != len(self.modes) or set(requested) != set(self.modes):
        raise ValueError(...)
    indices = [
        self.get_mode_index(mode) + offset
        for mode in requested
        for offset in (0, 1)
    ]
    displacement = self.displacement[indices].copy()
    covariance = self.covariance[np.ix_(indices, indices)].copy()
    return GaussianState(modes=requested, displacement=displacement, covariance=covariance)
```

`reorder_modes` explicitly changes only the *representation*, not the physical state — useful when two states from different circuits (e.g. before merging two `Circuit` layouts, Chapter 8) need to be brought into a common mode order.

---


=== Literature
The preparation of coherent, squeezed, and EPR states and the phase-space/Fock-space bridge are treated in depth in:

- #link("https://www.routledge.com/Quantum-Continuous-Variables-A-Primer-of-Theoretical-Methods/Serafini/p/book/9781032157238")[A. Serafini, *Quantum Continuous Variables: A Primer of Theoretical Methods*, 2nd ed. (CRC Press, 2023).]
- #link("https://doi.org/10.1103/RevModPhys.84.621")[C. Weedbrook et al., “Gaussian quantum information,” *Reviews of Modern Physics* 84, 621–669 (2012).]
- #link("https://doi.org/10.1140/epjst/e2012-01532-4")[S. Olivares, “Quantum optics in the phase space: A tutorial on Gaussian states,” *EPJ Special Topics* 203, 3–24 (2012).]
- #link("https://doi.org/10.1002/3527602976.ch4")[W. P. Schleich, “Quantum States in Phase Space,” in *Quantum Optics in Phase Space* (Wiley-VCH, 2001).]

Serafini is an advanced reference for Williamson decomposition, symplectic spectra, Gaussian operations, and the relation between phase-space and Hilbert-space descriptions.

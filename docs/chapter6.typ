#import "@preview/physica:0.9.8": *
#import "links.typ": src-link

// ==========================================
// CHAPTER 6
// ==========================================
= Chapter 6: Phase-Space Diagnostics & Entanglement Witnesses

The free functions at the end of #src-link("src/catsy/gaussian/__init__.py", label: [`gaussian/__init__.py`]) (`compute_wigner_analytically`, `compute_joint_correlation`, `compute_duan_inseparability`) form the *analysis layer* of the toolkit. They extract directly observable or certifying quantities from $(d, V)$ without needing to go through the Fock space of Chapter 5 — a key efficiency advantage of the Gaussian layer. Their plotting companions live in a separate module, #src-link("src/catsy/gaussian/visualization.py", label: [`gaussian/visualization.py`]), covered at the end of this chapter.

== Analytical Wigner function

For a Gaussian state, the Wigner quasi-probability distribution is itself a Gaussian function in phase space and is therefore non-negative. Negativity is a non-Gaussian feature of Wigner representations; the Gaussian case is reviewed by #link("https://doi.org/10.1140/epjst/e2012-01532-4")[Olivares (2012)] and #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)]. For a single mode with local displacement vector $d$ and local covariance $V$, it holds exactly that:
$ W(x, p) = 1 / (2 pi sqrt(det V)) exp(-1/2 (r - d)^T V^(-1) (r - d)), quad r = vec(x, p) $

`compute_wigner_analytically` evaluates this formula directly on a grid, without constructing any Fock-space representation:

```python
def compute_wigner_analytically(
    state: GaussianState, mode_name: str, x_max: float = 4.0, num_points: int = 150
):
    idx = state.get_mode_index(mode_name)
    d_mode = state.displacement[idx : idx + 2]
    V_mode = state.covariance[idx : idx + 2, idx : idx + 2]

    xvec = np.linspace(-x_max, x_max, num_points)
    pvec = np.linspace(-x_max, x_max, num_points)
    X, P = np.meshgrid(xvec, pvec)

    det_V = np.linalg.det(V_mode)
    inv_V = np.linalg.inv(V_mode)

    dX = X - d_mode[0]
    dP = P - d_mode[1]
    exponent = (
        dX * inv_V[0, 0] * dX
        + dX * inv_V[0, 1] * dP
        + dP * inv_V[1, 0] * dX
        + dP * inv_V[1, 1] * dP
    )
    W = (1.0 / (2.0 * np.pi * np.sqrt(det_V))) * np.exp(-0.5 * exponent)
    return W, X, P, mode_name
```

Since only the $2 times 2$ submatrix of the target mode is extracted, this evaluation is $O(1)$ per grid point, independent of the total number of modes — unlike a Wigner function reconstructed from a truncated Fock density matrix. Callers rarely need to invoke `compute_wigner_analytically` directly: `plot_wigner(state, mode_name)` in #src-link("src/catsy/gaussian/visualization.py", label: [`gaussian/visualization.py`]) computes this grid internally and renders it as a heatmap (diverging red-blue color scale, symmetric around zero) with white contour lines and the state's mean marked, titled with the mode name.

== Joint quadrature correlations

While the covariance matrix $V$ already encodes correlations between two modes numerically ($V_(i j)$), `compute_joint_correlation` makes this correlation *visible*: it computes the joint probability density of the same quadrature ($q$ or $p$) across two modes as a bivariate Gaussian over the reduced $2 times 2$ covariance matrix $V_"sub"$ of the selected quadrature components:

```python
idx_a = state.get_mode_index(mode_a) + offset
idx_b = state.get_mode_index(mode_b) + offset

V_sub = np.array([
    [state.covariance[idx_a, idx_a], state.covariance[idx_a, idx_b]],
    [state.covariance[idx_b, idx_a], state.covariance[idx_b, idx_b]],
])
```

where `offset = 0` selects the $q$ quadrature and `offset = 1` selects $p$. For the EPR pair constructed in Chapter 5, the $q$ correlation shows a distribution stretched along the diagonal $q_a = q_b$ (positive correlation), while the $p$ correlation is stretched along $p_a = -p_b$ (anti-correlation) — the visual signature of the variance formula given in Chapter 5, $"Var"(q_a - q_b) = "Var"(p_a + p_b) = e^(-2r)$.

== The Duan-Simon entanglement witness

Both the covariance matrix and the correlation plots show *correlation*, but on their own do not prove *entanglement*: purely classically correlated noise (e.g. `LossChannels.correlated_thermal_noise`) also produces visible correlations. `compute_duan_inseparability` implements the Duan-Giedke-Cirac-Zoller inseparability criterion. For arbitrary bipartite continuous-variable states it supplies a sufficient entanglement condition; for the two-mode Gaussian states relevant to this Gaussian layer, the criterion is necessary and sufficient. The original result is #link("https://doi.org/10.1103/PhysRevLett.84.2722")[Duan et al., *Physical Review Letters* 84, 2722 (2000)].

$ "Var"(q_a - q_b) + "Var"(p_a + p_b) >= 2 quad "(every separable state)" $

In the convention used by the toolkit (vacuum $= 1/2$), two independent vacua exactly saturate this bound at $2$ (`DUAN_SEPARABILITY_BOUND` in `core.py`). A measured value *strictly below* $2$ is a *sufficient* condition for non-classical entanglement between `mode_a` and `mode_b` — no classical correlation can undercut this bound, only an entangling operation such as the beam splitter used in `GaussianState.tmsv` can. For the two-mode Gaussian states considered here, that last caveat should be read carefully: the Duan criterion is equivalent to the Gaussian PPT/separability test. Simon’s independent phase-space formulation of the PPT criterion gives the corresponding necessary-and-sufficient separability condition for bipartite Gaussian states. For non-Gaussian states, the simple Duan test is only a sufficient witness.

```python
def compute_duan_inseparability(
    state: GaussianState, mode_a: str, mode_b: str
) -> float:
    idx_a = state.get_mode_index(mode_a)
    idx_b = state.get_mode_index(mode_b)
    V = state.covariance

    var_x_diff = V[idx_a, idx_a] + V[idx_b, idx_b] - 2 * V[idx_a, idx_b]
    var_p_sum = (
        V[idx_a + 1, idx_a + 1] + V[idx_b + 1, idx_b + 1] + 2 * V[idx_a + 1, idx_b + 1]
    )
    return float(var_x_diff + var_p_sum)
```

The signs of the cross terms directly mirror the structure of the EPR state: the minus sign in front of $V_(a b)$ in the $q$ term expects *positive* correlation ($q_a approx q_b$), while the plus sign in the $p$ term expects *negative* correlation ($p_a approx -p_b$) — exactly the statistics that `GaussianState.tmsv` constructs.

== Visualizing states and dynamics

Each analysis function above has a plotting companion in #src-link("src/catsy/gaussian/visualization.py", label: [`gaussian/visualization.py`]) that renders it as a Matplotlib figure. All of them return the figure without calling `plt.show()` unless passed `show=True`, and accept an optional `ax` so they can be composed into a larger layout (see the dashboards below).

*Single-state views* -- `plot_phase_space` (mean and $n_sigma$ uncertainty ellipse), `plot_wigner` (the heatmap described above), `plot_covariance_matrix` (the raw quadrature covariance as a heatmap), and `plot_mode_correlation_map` (the same covariance normalized to $[-1, 1]$, with mode boundaries drawn in, so cross-mode structure is visible independent of each mode's absolute variance).

*Dynamics* -- `plot_phase_space_trajectory` traces the mean through phase space over a sequence of states, with sparse uncertainty ellipses along the path; `plot_phase_space_trajectory_timecoded` is the same idea with the trajectory colored by an explicit `times` sequence (or step index) instead of drawn as a single flat line. `plot_covariance_evolution` and `plot_diagnostics` (purity, symplectic eigenvalues) track scalar summaries over the same sequence. `animate_phase_space` produces a `matplotlib.animation.FuncAnimation` rather than a static figure.

*Composite dashboards* -- `plot_state_dashboard` combines the single-state views above into one figure for a state (adding `plot_mode_correlation_map` automatically when the state has more than one mode); `plot_evolution` is the equivalent for a time sequence, combining the trajectory, covariance evolution, diagnostics, and one `plot_wigner` snapshot (the last of `wigner_indices`, default the final state) into one figure -- for several Wigner snapshots side by side instead, use `plot_wigner_evolution` directly. `plot_multimode_evolution` extends the trajectory view to several modes at once, with a shared panel tracking the strongest cross-mode correlation over time.

The Fock-space counterparts live in the sibling module, #src-link("src/catsy/fock/visualization.py", label: [`fock/visualization.py`]), and expose quantities invisible to the Gaussian $(d, V)$ description. The four public plotting functions share a simple contract: each returns a `matplotlib.figure.Figure`, does not display it unless `show=True`, and accepts `mode_idx` for selecting a mode from a multimode QuTiP state. They reduce the selected mode to a single-mode density matrix before plotting.

*Photon statistics* -- `plot_photon_statistics(rho, *, mode_idx=0, ax=None, n_max=None, show=False)` plots the photon-number probabilities and annotates the zero-delay second-order correlation $g^((2))(0)$. If `n_max` is omitted, the displayed cutoff is inferred from the occupied photon-number support; otherwise it selects the requested upper photon-number limit. The function returns the completed figure and can draw into an existing axis through `ax`.

*Fock density matrix* -- `plot_fock_density_matrix(rho, *, mode_idx=0, axes=None, show=False)` renders the magnitude and phase of the selected Fock-basis density matrix. `axes` may supply the pair of Matplotlib axes used for the two views. Near-zero matrix elements are masked in the phase view so that numerical phase noise is not presented as physical phase structure.

*Fock Wigner function* -- `plot_wigner(rho, *, mode_idx=0, xlim=(-5, 5), resolution=180, ax=None, projection="2d", show=False)` computes the Wigner function through QuTiP and renders either a 2D phase-space plot or a 3D surface. `xlim` defines the symmetric phase-space extent and `resolution` controls the grid density; `resolution` must be at least 20. `projection` accepts only `"2d"` or `"3d"`. In 3D mode, the surface colors are normalized to the actual minimum and maximum of the computed Wigner function, which preserves the visual distinction between positive and negative regions.

*Fock dashboard* -- `plot_fock_dashboard(rho, *, mode_idx=0, xlim=(-5, 5), resolution=140, n_max=None, show=False)` combines four complementary views: photon-number statistics, Wigner function, density-matrix magnitude, and density-matrix phase. It uses the same `mode_idx`, `xlim`, `resolution`, and `n_max` conventions as the component plots and returns one composite figure. The plotting functions derive concise state-aware descriptions (for example, Fock, vacuum, Poissonian, parity, or nonclassical descriptions) from the photon statistics, so titles identify the displayed state rather than only a mode index.

Both visualization modules share their figure-lifecycle and phase-space-styling primitives (`finalize_figure`, `style_phase_axes`) from #src-link("src/catsy/visualization.py", label: [`visualization.py`]), which keeps mixed Gaussian/Fock dashboards visually consistent.

---


=== Literature
The diagnostics in this chapter connect directly to the standard literature on Gaussian phase space and continuous-variable entanglement cited in Chapter 1 (Weedbrook et al. 2012), plus specifically:

- #link("https://doi.org/10.1140/epjst/e2012-01532-4")[S. Olivares, “Quantum optics in the phase space: A tutorial on Gaussian states,” *EPJ Special Topics* 203, 3–24 (2012).]
- #link("https://doi.org/10.1103/PhysRevLett.84.2722")[L.-M. Duan, G. Giedke, J. I. Cirac, and P. Zoller, “Inseparability criterion for continuous variable systems,” *Physical Review Letters* 84, 2722–2725 (2000).]
- #link("https://doi.org/10.1103/PhysRevLett.84.2726")[R. Simon, “Peres-Horodecki separability criterion for continuous variable systems,” *Physical Review Letters* 84, 2726–2729 (2000).]

The Duan and Simon papers are especially important here because they justify the entanglement interpretation of the numerical witness rather than merely providing a heuristic correlation measure.

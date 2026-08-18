#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 6
// ==========================================
= Chapter 6: Phase-Space Diagnostics & Entanglement Witnesses

The free functions at the end of `gaussian.py` (`compute_wigner_analytically`, `compute_joint_correlation`, `compute_duan_inseparability`, and their plotting companions) form the *analysis layer* of the toolkit. They extract directly observable or certifying quantities from $(d, V)$ without needing to go through the Fock space of Chapter 5 — a key efficiency advantage of the Gaussian layer.

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
    return W, X, P
```

Since only the $2 times 2$ submatrix of the target mode is extracted, this evaluation is $O(1)$ per grid point, independent of the total number of modes — unlike a Wigner function reconstructed from a truncated Fock density matrix. `plot_wigner` visualizes the result as a filled contour map with a diverging (red-blue) color scale, symmetric around the zero of the probability density.

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

In the convention used by the toolkit (vacuum $= 1/2$), two independent vacua exactly saturate this bound at $2$ (`DUAN_SEPARABILITY_BOUND` in `core.py`). A measured value *strictly below* $2$ is a *sufficient* condition for non-classical entanglement between `mode_a` and `mode_b` — no classical correlation can undercut this bound, only an entangling operation such as the beam splitter in `create_epr_pair` can. For the two-mode Gaussian states considered here, that last caveat should be read carefully: the Duan criterion is equivalent to the Gaussian PPT/separability test. Simon’s independent phase-space formulation of the PPT criterion gives the corresponding necessary-and-sufficient separability condition for bipartite Gaussian states. For non-Gaussian states, the simple Duan test is only a sufficient witness.

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
    return var_x_diff + var_p_sum
```

The signs of the cross terms directly mirror the structure of the EPR state: the minus sign in front of $V_(a b)$ in the $q$ term expects *positive* correlation ($q_a approx q_b$), while the plus sign in the $p$ term expects *negative* correlation ($p_a approx -p_b$) — exactly the statistics that `create_epr_pair` constructs.

---


== Scientific literature
The diagnostics in this chapter connect directly to the standard literature on Gaussian phase space and continuous-variable entanglement cited in Chapter 1 (Weedbrook et al. 2012), plus specifically:

- #link("https://doi.org/10.1140/epjst/e2012-01532-4")[S. Olivares, “Quantum optics in the phase space: A tutorial on Gaussian states,” *EPJ Special Topics* 203, 3–24 (2012).]
- #link("https://doi.org/10.1103/PhysRevLett.84.2722")[L.-M. Duan, G. Giedke, J. I. Cirac, and P. Zoller, “Inseparability criterion for continuous variable systems,” *Physical Review Letters* 84, 2722–2725 (2000).]
- #link("https://doi.org/10.1103/PhysRevLett.84.2726")[R. Simon, “Peres-Horodecki separability criterion for continuous variable systems,” *Physical Review Letters* 84, 2726–2729 (2000).]

The Duan and Simon papers are especially important here because they justify the entanglement interpretation of the numerical witness rather than merely providing a heuristic correlation measure.

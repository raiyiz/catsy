"""Gaussian-state simulation and analysis.

This module contains the continuous-variable Gaussian layer: states and
standard operations, general Gaussian channels, circuits, measurements, and
phase-space diagnostics. QuTiP is a required core dependency of catsy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.linalg

from .core import (
    DUAN_SEPARABILITY_BOUND,
    TOL_PHYSICALITY,
    TOL_TRACE_WARN,
    _apply_gaussian_transform,
    _check_non_negative,
    _check_thermal_correlation,
    _check_unit_interval,
    _check_positive_int,
    _json_load,
    _json_save,
    _symplectic_form,
    _validate_finite_array,
    _validate_gaussian_channel,
    _validate_physical_covariance,
    _williamson_decomposition,
)

logger = logging.getLogger("catsy")



# ========================================================================
# Gaussian
# ========================================================================

def _qutip_passive_unitary(O: np.ndarray, a_ops: list[Any]):
    """Build a QuTiP unitary implementing an orthogonal symplectic O."""
    n_modes = len(a_ops)
    A = O[0::2, 0::2]
    B = O[0::2, 1::2]
    C = O[1::2, 0::2]
    D = O[1::2, 1::2]

    U = 0.5 * (A + D + 1j * (C - B))
    u, _, vh = np.linalg.svd(U)
    U = u @ vh

    h = 1j * scipy.linalg.logm(U)
    h = 0.5 * (h + h.conj().T)

    H = 0
    for i in range(n_modes):
        for j in range(n_modes):
            hij = h[i, j]
            if abs(hij) > TOL_PHYSICALITY:
                H += hij * a_ops[i].dag() * a_ops[j]

    if H == 0:
        return qt.tensor(*[qt.qeye(a.dims[0][0]) for a in a_ops])
    return (-1j * H).expm()


@dataclass
class GaussianState:
    """A multi-mode Gaussian state, fully described by (modes, d, V)."""

    modes: tuple[str, ...]
    displacement: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self._validate()

    def _validate(self):
        n_modes = len(self.modes)
        if len(set(self.modes)) != n_modes:
            raise ValueError(f"Duplicate mode names in {self.modes!r}.")
        expected_dim = 2 * n_modes
        if self.displacement.shape != (expected_dim,):
            raise ValueError(
                f"displacement must have shape ({expected_dim},), "
                f"got {self.displacement.shape}."
            )
        if self.covariance.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"covariance must have shape ({expected_dim}, {expected_dim}), "
                f"got {self.covariance.shape}."
            )

        _validate_finite_array(self.displacement, "displacement")
        _validate_physical_covariance(self.covariance)

    def get_mode_index(self, mode_name: str) -> int:
        if mode_name not in self.modes:
            raise ValueError(f"Mode '{mode_name}' is not present in this state.")
        return self.modes.index(mode_name) * 2

    def reorder_modes(self, modes: tuple[str, ...] | list[str]) -> GaussianState:
        """Return an equivalent state with quadratures arranged in ``modes`` order.

        The requested modes must contain exactly the same unique mode names as
        this state. Both the displacement vector and covariance matrix are
        permuted together, so this operation changes only the representation,
        not the physical state.
        """
        requested = tuple(modes)

        if len(requested) != len(self.modes) or set(requested) != set(self.modes):
            raise ValueError(
                "Requested mode order must contain exactly the state's modes; "
                f"state={self.modes!r}, requested={requested!r}."
            )

        if len(set(requested)) != len(requested):
            raise ValueError(f"Duplicate mode names in requested order: {requested!r}.")

        if requested == self.modes:
            return self.copy()

        indices = [
            self.get_mode_index(mode) + offset
            for mode in requested
            for offset in (0, 1)
        ]

        displacement = self.displacement[indices].copy()
        covariance = self.covariance[np.ix_(indices, indices)].copy()

        return GaussianState(
            modes=requested,
            displacement=displacement,
            covariance=covariance,
        )

    def __repr__(self) -> str:
        purity = 1.0 / (
            2.0 ** len(self.modes) * np.sqrt(max(np.linalg.det(self.covariance), 0.0))
        )
        return f"GaussianState(modes={self.modes}, purity~{purity:.3f})"

    def copy(self) -> GaussianState:
        return GaussianState(
            modes=self.modes,
            displacement=self.displacement.copy(),
            covariance=self.covariance.copy(),
        )

    # -- Fock-space bridge --------------------------------------------------

    def to_qutip(self, N_cutoff: int = 15):
        """Convert this Gaussian state to a truncated QuTiP density matrix.

        The conversion uses a numerically stable Williamson decomposition of the
        covariance matrix, followed by a polar decomposition of the resulting
        symplectic matrix.  The thermal Williamson modes are prepared with
        ``qutip.thermal_dm``; the passive part is implemented with a
        number-conserving quadratic Hamiltonian and the positive symplectic part
        with a quadratic quadrature Hamiltonian.  The displacement is applied
        with QuTiP's ``displace`` primitive.

        Williamson's decomposition is exact mathematically; this implementation
        verifies the reconstructed symplectic transformation and covariance to
        floating-point tolerance. The returned density matrix is nevertheless
        represented in a finite Fock-space cutoff, so the final phase-space to
        Hilbert-space conversion can still incur truncation error.
        """

        _check_positive_int(N_cutoff, "N_cutoff")
        n_modes = len(self.modes)

        # The covariance is already checked for physicality by GaussianState.
        symplectic_values, S, D = _williamson_decomposition(self.covariance)

        # A numerical sanity check here is useful because this routine is the
        # bridge between the phase-space and Hilbert-space representations.
        Omega = _symplectic_form(n_modes)
        symplectic_residual = np.max(np.abs(S @ Omega @ S.T - Omega))
        covariance_residual = np.max(np.abs(S @ D @ S.T - self.covariance))
        if symplectic_residual > 1e-8 or covariance_residual > 1e-8:
            raise RuntimeError(
                "Williamson decomposition failed numerical consistency checks: "
                f"symplectic residual={symplectic_residual:.3e}, "
                f"covariance residual={covariance_residual:.3e}."
            )

        # The Williamson diagonal state is a tensor product of thermal states.
        rho_list = [
            qt.thermal_dm(N_cutoff, max(float(nu) - 0.5, 0.0))
            for nu in symplectic_values
        ]
        rho = qt.tensor(*rho_list)

        # Build canonical quadrature operators in the same interleaved
        # (x1,p1,x2,p2,...) convention as the phase-space representation.
        a_ops = []
        for i in range(n_modes):
            op_list = [qt.qeye(N_cutoff) for _ in range(n_modes)]
            op_list[i] = qt.destroy(N_cutoff)
            a_ops.append(qt.tensor(*op_list))

        r_ops = []
        for a in a_ops:
            r_ops.append((a + a.dag()) / np.sqrt(2.0))
            r_ops.append((a - a.dag()) / (1j * np.sqrt(2.0)))

        # Polar decomposition of the symplectic transformation:
        #     S = P O
        # with P positive symplectic and O orthogonal symplectic.
        # Unlike logm(S) directly, logm(P) is the logarithm of a positive
        # matrix and therefore gives the well-behaved quadratic generator we
        # need.  O is a passive Gaussian transformation.
        P = scipy.linalg.sqrtm(S @ S.T).real
        P_inv = scipy.linalg.inv(P)
        O = P_inv @ S

        passive_residual = max(
            np.max(np.abs(O.T @ O - np.eye(2 * n_modes))),
            np.max(np.abs(O @ Omega @ O.T - Omega)),
        )
        positive_residual = np.max(np.abs(P @ Omega @ P.T - Omega))
        if passive_residual > 1e-8 or positive_residual > 1e-8:
            raise RuntimeError(
                "Symplectic polar decomposition failed numerical consistency "
                f"checks: passive residual={passive_residual:.3e}, "
                f"positive residual={positive_residual:.3e}."
            )

        # Apply the passive transformation first, then the positive one, so
        # the total phase-space transformation is P @ O = S.
        U_passive = _qutip_passive_unitary(O, a_ops)
        rho = U_passive * rho * U_passive.dag()

        log_P = scipy.linalg.logm(P).real
        G = -Omega @ log_P
        G = 0.5 * (G + G.T)

        H_positive = 0
        for i in range(2 * n_modes):
            for j in range(2 * n_modes):
                gij = G[i, j]
                if np.abs(gij) > TOL_PHYSICALITY:
                    H_positive += 0.5 * gij * r_ops[i] * r_ops[j]

        if H_positive != 0:
            U_positive = (-1j * H_positive).expm()
            rho = U_positive * rho * U_positive.dag()

        # QuTiP's displacement convention is alpha=(dx+i*dp)/sqrt(2), which
        # directly matches the x,p convention used by GaussianState.
        for i in range(n_modes):
            dx = float(self.displacement[2 * i])
            dp = float(self.displacement[2 * i + 1])
            if abs(dx) <= TOL_PHYSICALITY and abs(dp) <= TOL_PHYSICALITY:
                continue

            alpha = (dx + 1j * dp) / np.sqrt(2.0)
            op_list = [qt.qeye(N_cutoff) for _ in range(n_modes)]
            op_list[i] = qt.displace(N_cutoff, alpha)
            D_op = qt.tensor(*op_list)
            rho = D_op * rho * D_op.dag()

        return rho

    # -- Plotting -------------------------------------------------------

    def plot_covariance(self):
        """Visualize correlations between all registered modes."""
        ticks = []
        for m in self.modes:
            ticks.extend([f"q_{m}", f"p_{m}"])

        plt.figure(figsize=(6, 5))
        im = plt.imshow(self.covariance, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Variance / covariance")
        plt.xticks(range(len(ticks)), ticks)
        plt.yticks(range(len(ticks)), ticks)
        plt.title("Multi-mode covariance matrix V")

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": list(self.modes),
            "displacement": self.displacement.tolist(),
            "covariance": self.covariance.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianState:
        return cls(
            modes=tuple(data["modes"]),
            displacement=np.array(data["displacement"], dtype=float),
            covariance=np.array(data["covariance"], dtype=float),
        )

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GaussianState:
        return cls.from_dict(_json_load(path))


class GaussianOperations:
    @staticmethod
    def create_vacuum(modes: tuple[str, ...]) -> GaussianState:
        """Multi-mode vacuum state (V = 0.5 * I)."""
        dim = 2 * len(modes)
        d = np.zeros(dim)
        V = 0.5 * np.eye(dim)
        return GaussianState(modes=modes, displacement=d, covariance=V)

    @staticmethod
    def create_coherent(
        modes: tuple[str, ...], alphas: complex | Sequence[complex]
    ) -> GaussianState:
        """Multi-mode coherent state |alpha_1> ⊗ ... ⊗ |alpha_n> -- a vacuum
        with each mode displaced by its complex amplitude alpha_k. Passing a
        single scalar broadcasts the same alpha to every mode."""
        if np.isscalar(alphas):
            alphas = [alphas] * len(modes)
        alphas = list(alphas)
        if len(alphas) != len(modes):
            raise ValueError(
                f"Got {len(alphas)} alpha(s) for {len(modes)} mode(s); "
                "pass one alpha per mode (or a single scalar to broadcast)."
            )

        state = GaussianOperations.create_vacuum(modes)
        for mode, alpha in zip(modes, alphas):
            state = GaussianOperations.apply_displacement(state, mode, alpha)
        return state

    @staticmethod
    def create_epr_pair(mode_a: str, mode_b: str, r: float) -> GaussianState:
        """Canonical two-mode squeezed vacuum (EPR pair): squeeze `mode_a` in
        x, `mode_b` in p, then combine them on a 50:50 beam splitter. This is
        the standard recipe for *genuine* (non-classical) CV entanglement --
        as opposed to merely correlated noise from a channel, which can look
        similar in a scatter plot but never violates the Duan-Simon bound
        (see `compute_duan_inseparability`).

        Produces Var(x_a - x_b) = Var(p_a + p_b) = exp(-2r): x_a and x_b end
        up positively correlated, p_a and p_b end up anti-correlated, and
        both combined variances drop below the vacuum (shot-noise) level for
        any r > 0.
        """
        _check_non_negative(r, "r")
        state = GaussianOperations.create_vacuum((mode_a, mode_b))
        state = GaussianOperations.apply_squeezing(state, mode=mode_a, r=r, theta=0.0)
        state = GaussianOperations.apply_squeezing(
            state, mode=mode_b, r=r, theta=np.pi / 2
        )
        return GaussianOperations.apply_beam_splitter(
            state, mode_a=mode_a, mode_b=mode_b, eta=0.5
        )

    @staticmethod
    def apply_squeezing(
        state: GaussianState, mode: str, r: float, theta: float = 0.0
    ) -> GaussianState:
        """Single-mode squeezing on `mode` (squeeze strength r, phase theta)."""
        idx = state.get_mode_index(mode)
        dim = len(state.displacement)

        S_local = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S_local = R @ S_local @ R.T

        S_global = np.eye(dim)
        S_global[idx : idx + 2, idx : idx + 2] = S_local

        return _apply_gaussian_transform(state, S_global)

    @staticmethod
    def apply_phase_rotation(
        state: GaussianState, mode: str, phi: float
    ) -> GaussianState:
        """Phase-space rotation by angle `phi` on `mode` (a passive, energy-
        preserving gate -- the piece needed alongside squeezing + beam
        splitters to generate arbitrary single- and two-mode Gaussian
        unitaries)."""
        idx = state.get_mode_index(mode)
        dim = len(state.displacement)

        R_local = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        R_global = np.eye(dim)
        R_global[idx : idx + 2, idx : idx + 2] = R_local

        return _apply_gaussian_transform(state, R_global)

    @staticmethod
    def apply_displacement(
        state: GaussianState,
        mode: str,
        alpha: complex | None = None,
        *,
        x: float | None = None,
        p: float | None = None,
    ) -> GaussianState:
        """Phase-space displacement D(alpha) = exp(alpha*adag - alpha^*a) on
        `mode`: shifts the mean by (x, p) = (sqrt(2)*Re(alpha), sqrt(2)*Im(alpha))
        and leaves the covariance untouched (displacement is affine, not
        symplectic-mixing, so it never changes purity/entanglement).

        Give either `alpha` (complex amplitude) or both `x` and `p`
        (quadrature shifts directly) -- not both.
        """
        if alpha is not None and (x is not None or p is not None):
            raise ValueError("Pass either `alpha` or (`x`, `p`), not both.")
        if alpha is not None:
            d_x, d_p = np.sqrt(2.0) * np.real(alpha), np.sqrt(2.0) * np.imag(alpha)
        elif x is not None and p is not None:
            d_x, d_p = x, p
        else:
            raise ValueError("Must supply either `alpha` or both `x` and `p`.")

        idx = state.get_mode_index(mode)
        new_d = state.displacement.copy()
        new_d[idx] += d_x
        new_d[idx + 1] += d_p
        return GaussianState(
            modes=state.modes, displacement=new_d, covariance=state.covariance.copy()
        )

    @staticmethod
    def apply_beam_splitter(
        state: GaussianState, mode_a: str, mode_b: str, eta: float
    ) -> GaussianState:
        """Lossless beam splitter with power transmissivity eta on (mode_a, mode_b)."""
        if mode_a == mode_b:
            raise ValueError("mode_a and mode_b must be different modes.")
        _check_unit_interval(eta, "eta")

        idx_a = state.get_mode_index(mode_a)
        idx_b = state.get_mode_index(mode_b)
        dim = len(state.displacement)

        t = np.sqrt(eta)
        r_coeff = np.sqrt(1 - eta)

        S_BS = np.eye(dim)
        I2 = np.eye(2)
        S_BS[idx_a : idx_a + 2, idx_a : idx_a + 2] = t * I2
        S_BS[idx_a : idx_a + 2, idx_b : idx_b + 2] = r_coeff * I2
        S_BS[idx_b : idx_b + 2, idx_a : idx_a + 2] = -r_coeff * I2
        S_BS[idx_b : idx_b + 2, idx_b : idx_b + 2] = t * I2

        return _apply_gaussian_transform(state, S_BS)

    @staticmethod
    def apply_loss(state: GaussianState, mode: str, eta: float) -> GaussianState:
        """Vacuum-coupled loss (transmissivity eta) on `mode`."""
        _check_unit_interval(eta, "eta")
        idx = state.get_mode_index(mode)
        dim = len(state.displacement)

        X = np.eye(dim)
        X[idx : idx + 2, idx : idx + 2] = np.sqrt(eta) * np.eye(2)

        Y = np.zeros((dim, dim))
        Y[idx : idx + 2, idx : idx + 2] = (1 - eta) * 0.5 * np.eye(2)

        return _apply_gaussian_transform(state, X, noise=Y)

# ========================================================================
# Channels
# ========================================================================

@dataclass
class GaussianChannel:
    """A general Gaussian channel d' = X@d + d0, V' = X@V@X.T + Y acting on
    a subset of modes."""

    target_modes: tuple[str, ...]
    X: np.ndarray
    Y: np.ndarray
    d0: np.ndarray

    def __post_init__(self):
        if len(set(self.target_modes)) != len(self.target_modes):
            raise ValueError(f"Duplicate target mode names in {self.target_modes!r}.")

        dim = 2 * len(self.target_modes)
        _validate_gaussian_channel(
            self.X, self.Y, self.d0, expected_dim=dim
        )

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

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_modes": list(self.target_modes),
            "X": self.X.tolist(),
            "Y": self.Y.tolist(),
            "d0": self.d0.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianChannel:
        return cls(
            target_modes=tuple(data["target_modes"]),
            X=np.array(data["X"], dtype=float),
            Y=np.array(data["Y"], dtype=float),
            d0=np.array(data["d0"], dtype=float),
        )

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GaussianChannel:
        return cls.from_dict(_json_load(path))


class LossChannels:
    """Factory for standard optical noise channels."""

    @staticmethod
    def thermal_loss(mode: str, eta: float, n_thermal: float) -> GaussianChannel:
        """Loss (transmissivity eta) combined with thermal environment noise."""
        _check_unit_interval(eta, "eta")
        _check_non_negative(n_thermal, "n_thermal")

        X = np.sqrt(eta) * np.eye(2)
        V_env = (n_thermal + 0.5) * np.eye(2)
        Y = (1 - eta) * V_env
        d0 = np.zeros(2)
        return GaussianChannel(target_modes=(mode,), X=X, Y=Y, d0=d0)

    @staticmethod
    def classical_phase_jitter(mode: str, sigma_phi: float) -> GaussianChannel:
        """Small-angle approximation of phase jitter: extra noise added to the
        p-quadrature only, proportional to the jitter variance."""
        _check_non_negative(sigma_phi, "sigma_phi")
        X = np.eye(2)
        Y = np.diag(
            [0.0, sigma_phi**2]
        )  # fixed: was shape (1,2), invalid for a 2x2 channel
        d0 = np.zeros(2)
        return GaussianChannel(target_modes=(mode,), X=X, Y=Y, d0=d0)

    @staticmethod
    def correlated_thermal_noise(
        mode_a: str, mode_b: str, eta: float, n_thermal: float, c_correlation: float
    ) -> GaussianChannel:
        """Correlated thermal noise on two modes coupled to the same bath."""
        _check_unit_interval(eta, "eta")
        _check_non_negative(n_thermal, "n_thermal")
        _check_thermal_correlation(c_correlation, n_thermal)

        X = np.sqrt(eta) * np.eye(4)
        V_diag = (1 - eta) * (n_thermal + 0.5) * np.eye(2)
        V_cross = (1 - eta) * c_correlation * np.eye(2)
        Y = np.block([[V_diag, V_cross], [V_cross.T, V_diag]])
        d0 = np.zeros(4)
        return GaussianChannel(target_modes=(mode_a, mode_b), X=X, Y=Y, d0=d0)

# ========================================================================
# Circuit
# ========================================================================

logger = logging.getLogger("catsy")

@dataclass
class CircuitOperation:
    """One step in a compiled circuit: a registry key + its target modes and
    kwargs. Deliberately holds no function reference, so it stays trivially
    JSON-serializable."""

    name: str
    modes: tuple[str, ...]
    kwargs: dict[str, Any]


def _op_squeeze(
    state: GaussianState, modes: tuple[str, ...], **kwargs
) -> GaussianState:
    return GaussianOperations.apply_squeezing(state, mode=modes[0], **kwargs)


def _op_rotate(state: GaussianState, modes: tuple[str, ...], **kwargs) -> GaussianState:
    return GaussianOperations.apply_phase_rotation(state, mode=modes[0], **kwargs)


def _op_displace(
    state: GaussianState, modes: tuple[str, ...], **kwargs
) -> GaussianState:
    return GaussianOperations.apply_displacement(state, mode=modes[0], **kwargs)


def _op_beam_splitter(
    state: GaussianState, modes: tuple[str, ...], **kwargs
) -> GaussianState:
    return GaussianOperations.apply_beam_splitter(
        state, mode_a=modes[0], mode_b=modes[1], **kwargs
    )


def _op_loss(state: GaussianState, modes: tuple[str, ...], **kwargs) -> GaussianState:
    return GaussianOperations.apply_loss(state, mode=modes[0], **kwargs)


def _op_thermal_loss(
    state: GaussianState, modes: tuple[str, ...], **kwargs
) -> GaussianState:
    return LossChannels.thermal_loss(mode=modes[0], **kwargs).apply(state)


# Registry maps serialized operation names to their state-transforming functions.
OPERATION_REGISTRY: dict[str, Callable[..., GaussianState]] = {
    "Squeezing": _op_squeeze,
    "PhaseRotation": _op_rotate,
    "Displacement": _op_displace,
    "BeamSplitter": _op_beam_splitter,
    "Loss": _op_loss,
    "ThermalLossChannel": _op_thermal_loss,
}


@dataclass
class GaussianCircuit:
    """Sequences a chain of Gaussian gates/channels and runs them over a
    registered set of modes."""

    modes: tuple[str, ...] = field(default_factory=tuple)
    _operations: list[CircuitOperation] = field(default_factory=list, init=False)
    # Per-mode starting amplitude (0 -> vacuum), used by compile_and_run when
    # no initial_state is supplied -- see add_mode(alpha=...).
    _initial_alphas: dict[str, complex] = field(default_factory=dict, init=False)

    @classmethod
    def register(cls, name: str, fn: Callable[..., GaussianState]) -> None:
        """Register a new circuit-operation kind so `.compile_and_run` can execute it."""
        OPERATION_REGISTRY[name] = fn

    def add_mode(self, mode_name: str, alpha: complex = 0.0) -> GaussianCircuit:
        """Register a new mode, optionally starting it in the coherent state
        |alpha> instead of vacuum (only takes effect when `compile_and_run`
        is called without an explicit `initial_state`)."""
        if mode_name in self.modes:
            raise ValueError(
                f"Mode '{mode_name}' is already registered in this circuit."
            )
        self.modes = self.modes + (mode_name,)
        self._initial_alphas[mode_name] = alpha
        return self

    def _add_op(self, name: str, modes: tuple[str, ...], **kwargs) -> GaussianCircuit:
        self._operations.append(CircuitOperation(name=name, modes=modes, kwargs=kwargs))
        return self

    def squeeze(self, mode: str, r: float, theta: float = 0.0) -> GaussianCircuit:
        return self._add_op("Squeezing", (mode,), r=r, theta=theta)

    def rotate(self, mode: str, phi: float) -> GaussianCircuit:
        return self._add_op("PhaseRotation", (mode,), phi=phi)

    def displace(
        self,
        mode: str,
        alpha: complex | None = None,
        *,
        x: float | None = None,
        p: float | None = None,
    ) -> GaussianCircuit:
        """Add a displacement gate. Give either `alpha` or both `x` and `p`;
        either way the op is stored as (x, p) so the circuit stays plain-
        float JSON-serializable (see `to_dict`)."""
        if alpha is not None and (x is not None or p is not None):
            raise ValueError("Pass either `alpha` or (`x`, `p`), not both.")
        if alpha is not None:
            x, p = np.sqrt(2.0) * np.real(alpha), np.sqrt(2.0) * np.imag(alpha)
        elif x is None or p is None:
            raise ValueError("Must supply either `alpha` or both `x` and `p`.")
        return self._add_op("Displacement", (mode,), x=float(x), p=float(p))

    def beam_splitter(self, mode_a: str, mode_b: str, eta: float) -> GaussianCircuit:
        return self._add_op("BeamSplitter", (mode_a, mode_b), eta=eta)

    def loss(self, mode: str, eta: float) -> GaussianCircuit:
        return self._add_op("Loss", (mode,), eta=eta)

    def thermal_loss(self, mode: str, eta: float, n_thermal: float) -> GaussianCircuit:
        return self._add_op("ThermalLossChannel", (mode,), eta=eta, n_thermal=n_thermal)

    def compile_and_run(
        self, initial_state: GaussianState | None = None
    ) -> GaussianState:
        """Validate every operation against the registered modes and run the
        chain sequentially."""
        if not self.modes:
            raise ValueError("Circuit has no registered modes.")

        if initial_state is None:
            alphas = [self._initial_alphas.get(m, 0.0) for m in self.modes]
            current_state = GaussianOperations.create_coherent(self.modes, alphas)
        else:
            if set(initial_state.modes) != set(self.modes):
                raise ValueError(
                    "Initial state's modes don't match the circuit's modes."
                )
            # Circuit order is canonical.  A state may arrive with the same
            # named modes in a different order; reorder it once at the boundary
            # so every subsequent operation sees the same positional layout.
            current_state = initial_state.reorder_modes(self.modes)

        logger.debug(
            "Running circuit over modes %s (%d ops)", self.modes, len(self._operations)
        )

        for idx, op in enumerate(self._operations):
            for m in op.modes:
                if m not in self.modes:
                    raise ValueError(
                        f"Op #{idx} ({op.name}): mode '{m}' is not registered in this circuit."
                    )
            if op.name not in OPERATION_REGISTRY:
                raise KeyError(
                    f"Op #{idx}: unknown operation '{op.name}'. "
                    f"Known ops: {sorted(OPERATION_REGISTRY)}."
                )
            current_state = OPERATION_REGISTRY[op.name](
                current_state, op.modes, **op.kwargs
            )
            logger.debug(
                "[%d/%d] applied %s on %s",
                idx + 1,
                len(self._operations),
                op.name,
                op.modes,
            )

        return current_state

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": list(self.modes),
            # Stored as [re, im] pairs -- complex isn't JSON-serializable.
            "initial_alphas": {
                m: [np.real(a), np.imag(a)] for m, a in self._initial_alphas.items()
            },
            "operations": [
                {"name": op.name, "modes": list(op.modes), "kwargs": op.kwargs}
                for op in self._operations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianCircuit:
        circuit = cls(modes=tuple(data["modes"]))
        circuit._initial_alphas = {
            m: complex(re, im) for m, (re, im) in data.get("initial_alphas", {}).items()
        }
        for op in data["operations"]:
            if op["name"] not in OPERATION_REGISTRY:
                raise KeyError(
                    f"Unknown operation '{op['name']}' in serialized circuit."
                )
            circuit._operations.append(
                CircuitOperation(
                    name=op["name"], modes=tuple(op["modes"]), kwargs=op["kwargs"]
                )
            )
        return circuit

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GaussianCircuit:
        return cls.from_dict(_json_load(path))

# ========================================================================
# Measurements
# ========================================================================

class GaussianMeasurements:
    @staticmethod
    def homodyne_measurement(
        state: GaussianState,
        measured_mode: str,
        phi: float,
        outcome: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, GaussianState]:
        """Perform an ideal single-quadrature homodyne measurement.

        The measured quadrature is rotated onto ``x`` and conditioned using
        the Gaussian Schur complement.  ``outcome`` may be supplied to force
        a particular measurement result (useful for deterministic tests).
        """
        if not np.isfinite(phi):
            raise ValueError(f"phi must be finite, got {phi!r}.")
        if outcome is not None:
            if not np.isscalar(outcome) or not np.isfinite(outcome):
                raise ValueError("homodyne outcome must be a finite scalar.")

        n_modes = len(state.modes)
        idx_m = state.get_mode_index(measured_mode)

        R_local = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        R_global = np.eye(2 * n_modes)
        R_global[idx_m : idx_m + 2, idx_m : idx_m + 2] = R_local

        d_rot = R_global @ state.displacement
        V_rot = R_global @ state.covariance @ R_global.T

        idx_x = idx_m
        remaining_indices = [
            i for i in range(2 * n_modes) if i != idx_x and i != idx_m + 1
        ]

        V_MM = float(V_rot[idx_x, idx_x])
        if not np.isfinite(V_MM) or V_MM <= TOL_PHYSICALITY:
            raise ValueError(
                "homodyne measurement variance must be finite and positive; "
                f"got {V_MM:.3e}."
            )

        V_MR = V_rot[idx_x, remaining_indices]
        V_RM = V_rot[remaining_indices, idx_x]
        V_RR = V_rot[np.ix_(remaining_indices, remaining_indices)]

        d_M = float(d_rot[idx_x])
        d_R = d_rot[remaining_indices]

        if outcome is None:
            rng = rng if rng is not None else np.random.default_rng()
            measured_value = float(rng.normal(loc=d_M, scale=np.sqrt(V_MM)))
        else:
            measured_value = float(outcome)

        gain = V_RM / V_MM
        d_cond = d_R + gain * (measured_value - d_M)
        V_cond = V_RR - np.outer(V_RM, V_MR) / V_MM
        V_cond = 0.5 * (V_cond + V_cond.T)

        remaining_modes = tuple(m for m in state.modes if m != measured_mode)
        return measured_value, GaussianState(remaining_modes, d_cond, V_cond)

    @staticmethod
    def heterodyne_measurement(
        state: GaussianState,
        measured_mode: str,
        outcome: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, GaussianState]:
        """Heterodyne (double-homodyne) measurement on `measured_mode`: both
        quadratures are measured simultaneously, equivalent to splitting the
        mode 50:50 against vacuum and homodyning each output. Unlike
        `homodyne_measurement`, the outcome is a 2-vector (x, p) and the
        measured mode collapses onto (approximately) a coherent state rather
        than a squeezed one, because the extra vacuum port contributes a
        fixed 0.5*I of measurement noise.
        """
        idx_m = state.get_mode_index(measured_mode)
        dim = len(state.displacement)
        remaining_indices = [i for i in range(dim) if i < idx_m or i > idx_m + 1]

        V_MM = state.covariance[idx_m : idx_m + 2, idx_m : idx_m + 2]
        V_MR = state.covariance[idx_m : idx_m + 2, remaining_indices]
        V_RM = V_MR.T
        V_RR = state.covariance[np.ix_(remaining_indices, remaining_indices)]
        d_M = state.displacement[idx_m : idx_m + 2]
        d_R = state.displacement[remaining_indices]

        V_eff = 0.5 * (V_MM + 0.5 * np.eye(2)) + 0.5 * (V_MM + 0.5 * np.eye(2)).T
        if not np.all(np.isfinite(V_eff)):
            raise ValueError("heterodyne effective covariance must be finite.")

        try:
            np.linalg.cholesky(V_eff)
        except np.linalg.LinAlgError as exc:
            raise ValueError("heterodyne effective covariance must be positive definite.") from exc

        if outcome is None:
            rng = rng if rng is not None else np.random.default_rng()
            measured_outcome = rng.multivariate_normal(mean=d_M, cov=V_eff)
        else:
            measured_outcome = np.asarray(outcome, dtype=float)
            if measured_outcome.shape != (2,):
                raise ValueError(
                    "heterodyne outcome must have shape (2,), "
                    f"got {measured_outcome.shape}."
                )
            if not np.all(np.isfinite(measured_outcome)):
                raise ValueError("heterodyne outcome must contain only finite values.")

        gain = np.linalg.solve(V_eff, V_MR).T
        innovation = measured_outcome - d_M
        d_cond = d_R + gain @ innovation
        V_cond = V_RR - gain @ V_MR
        V_cond = 0.5 * (V_cond + V_cond.T)

        remaining_modes = tuple(m for m in state.modes if m != measured_mode)
        return measured_outcome, GaussianState(remaining_modes, d_cond, V_cond)

# ========================================================================
# Analysis
# ========================================================================

def compute_wigner_analytically(
    state: GaussianState, mode_name: str, x_max: float = 4.0, num_points: int = 150
):
    """Wigner function of a single mode, computed analytically from (d, V) —
    no Hilbert-space truncation involved."""
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


def plot_wigner(W, X, P, mode_name: str):
    plt.figure(figsize=(6, 5))
    span = max(np.max(W), np.abs(np.min(W)))
    contour = plt.contourf(X, P, W, 100, cmap="RdBu_r", vmin=-span, vmax=span)
    plt.colorbar(contour, label="Wigner probability density")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.title(f"Wigner function for mode '{mode_name}'")
    plt.xlabel("x (position / in-phase quadrature)")
    plt.ylabel("p (momentum / quadrature phase)")
    plt.axis("equal")
    plt.show()


def compute_joint_correlation(
    state: GaussianState,
    mode_a: str,
    mode_b: str,
    x_max: float = 3.0,
    num_points: int = 150,
    quadrature: str = "x",
):
    """Joint probability distribution of the same quadrature on two modes
    (e.g. x_a vs x_b, or p_a vs p_b) -- the tool for actually *seeing* an
    EPR-style correlation or anti-correlation, as opposed to only reading it
    off the covariance matrix. `quadrature` selects which pair: 'x' shows
    position correlation, 'p' shows momentum correlation (anti-correlated,
    for the standard `GaussianOperations.create_epr_pair` construction).
    """
    if quadrature not in ("x", "p"):
        raise ValueError(f"quadrature must be 'x' or 'p', got {quadrature!r}.")
    offset = 0 if quadrature == "x" else 1

    idx_a = state.get_mode_index(mode_a) + offset
    idx_b = state.get_mode_index(mode_b) + offset

    V_sub = np.array(
        [
            [state.covariance[idx_a, idx_a], state.covariance[idx_a, idx_b]],
            [state.covariance[idx_b, idx_a], state.covariance[idx_b, idx_b]],
        ]
    )
    d_sub = np.array([state.displacement[idx_a], state.displacement[idx_b]])

    xvec = np.linspace(-x_max, x_max, num_points)
    X_a, X_b = np.meshgrid(xvec, xvec)

    det_V = np.linalg.det(V_sub)
    inv_V = np.linalg.inv(V_sub)

    dX_a = X_a - d_sub[0]
    dX_b = X_b - d_sub[1]
    exponent = (
        inv_V[0, 0] * dX_a**2
        + (inv_V[0, 1] + inv_V[1, 0]) * dX_a * dX_b
        + inv_V[1, 1] * dX_b**2
    )
    P = (1.0 / (2.0 * np.pi * np.sqrt(det_V))) * np.exp(-0.5 * exponent)
    return P, X_a, X_b


def plot_joint_correlation(
    P, X_a, X_b, mode_a: str, mode_b: str, quadrature: str = "x"
):
    plt.figure(figsize=(6, 5))
    plt.contourf(X_a, X_b, P, 100, cmap="viridis")
    plt.colorbar(label="Probability density")
    plt.title(f"Correlation: quadrature {quadrature}_{mode_a} vs {quadrature}_{mode_b}")
    plt.xlabel(f"{quadrature}_{mode_a}")
    plt.ylabel(f"{quadrature}_{mode_b}")
    plt.axis("equal")
    plt.show()


def compute_duan_inseparability(
    state: GaussianState, mode_a: str, mode_b: str
) -> float:
    """Duan-Simon two-mode entanglement witness (Duan, Giedke, Cirac & Zoller,
    PRL 84, 2722 (2000)). In this codebase's vacuum=0.5 convention, *every*
    separable (classically-correlated-at-best) state satisfies

        Var(x_a - x_b) + Var(p_a + p_b) >= DUAN_SEPARABILITY_BOUND (== 2.0),

    with two independent vacua exactly saturating the bound. A value
    strictly below it is a *sufficient* condition for genuine, non-classical
    entanglement between mode_a and mode_b: no amount of classical
    correlation (e.g. from `LossChannels.correlated_thermal_noise`) can beat
    it, only a genuinely entangling operation like the beam splitter in
    `GaussianOperations.create_epr_pair` can. The bound is not necessary --
    some entangled states pass it undetected -- so failing to beat it does
    not itself prove separability.
    """
    idx_a = state.get_mode_index(mode_a)
    idx_b = state.get_mode_index(mode_b)
    V = state.covariance

    var_x_diff = V[idx_a, idx_a] + V[idx_b, idx_b] - 2 * V[idx_a, idx_b]
    var_p_sum = (
        V[idx_a + 1, idx_a + 1] + V[idx_b + 1, idx_b + 1] + 2 * V[idx_a + 1, idx_b + 1]
    )
    return var_x_diff + var_p_sum

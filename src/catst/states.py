"""Continuous-variable (CV) Gaussian quantum optics toolkit.

Layered design:
  - Phase-space layer (GaussianState / GaussianOperations / GaussianChannel /
    GaussianCircuit): pure numpy + scipy, no Hilbert-space cost. This is where
    almost all circuit-building work happens.
  - Fock-space layer (to_qutip, NonGaussianOperations, QBSSimulator): only
    imports qutip lazily, on first actual use, so building and manipulating
    purely Gaussian circuits never pays for qutip's (slow) import or for
    allocating a Hilbert space you don't need.

Design notes for anyone extending this:
  - New circuit gates/channels are added via `GaussianCircuit.register`
    rather than by editing `compile_and_run` — see OPERATION_REGISTRY.
  - All public entry points validate their inputs and raise ValueError /
    KeyError with a specific message; nothing relies on `assert`, which
    disappears under `python -O`.
  - Progress/diagnostic messages go through the `logging` module at DEBUG
    level rather than `print`, so they cost nothing unless a caller opts in
    with `logging.getLogger("catst").setLevel(logging.DEBUG)`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

logger = logging.getLogger("catst")

# Centralized numerical tolerances (previously scattered magic numbers).
TOL_SYMPLECTIC_FLOOR = (
    0.499  # symplectic eigenvalues below this are clamped to vacuum (0.5)
)
TOL_ZERO_ENTRY = (
    1e-9  # negligible matrix-entry threshold when building qutip Hamiltonians
)
TOL_ZERO_TRACE = (
    1e-11  # a heralded/conditioned state with trace below this is unphysical
)
TOL_TRACE_WARN = (
    1e-6  # deviation from tr(rho) == 1 worth warning about (truncation error)
)


def _lazy_qutip():
    """Import qutip on first use only, so pure phase-space work never pays for it."""
    import qutip as qt

    return qt


def _lazy_pyplot():
    """Import matplotlib.pyplot on first use only."""
    import matplotlib.pyplot as plt

    return plt


def _check_unit_interval(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")


def _check_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


# ---------------------------------------------------------------------------
# Phase-space state
# ---------------------------------------------------------------------------


@dataclass
class GaussianState:
    """A multi-mode Gaussian state, fully described by (modes, d, V)."""

    modes: tuple[str, ...]
    displacement: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
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

    def get_mode_index(self, mode_name: str) -> int:
        if mode_name not in self.modes:
            raise ValueError(f"Mode '{mode_name}' is not present in this state.")
        return self.modes.index(mode_name) * 2

    def copy(self) -> GaussianState:
        return GaussianState(
            modes=self.modes,
            displacement=self.displacement.copy(),
            covariance=self.covariance.copy(),
        )

    # -- Fock-space bridge --------------------------------------------------

    def to_qutip(self, N_cutoff: int = 15):
        """Exact conversion of (d, V) into a QuTiP density matrix via the
        Williamson decomposition (thermal states + a symplectic unitary).

        Known limitation: the symplectic generator is recovered via
        `scipy.linalg.sqrtm`/`logm` on the covariance matrix, which can lose
        a fraction of a percent of trace fidelity for some covariance
        matrices (this is a numerical property of that matrix decomposition,
        not a Fock-space truncation effect — increasing N_cutoff will not
        fix it). If `to_qutip` warns about trace deviation, check whether
        raising N_cutoff helps first; if the deviation is cutoff-independent,
        it's this decomposition's precision limit for that particular V.
        """
        qt = _lazy_qutip()
        n_modes = len(self.modes)

        # Fundamental symplectic form Omega = bigoplus [[0,1],[-1,0]]
        omega_1 = np.array([[0, 1], [-1, 0]])
        Omega = scipy.linalg.block_diag(*[omega_1 for _ in range(n_modes)])

        # Symplectic eigenvalues from i*Omega*V; eigenvalues occur in +-nu pairs.
        M = 1j * Omega @ self.covariance
        eigvals = np.linalg.eigvals(M)
        nu = sorted(np.abs(eigvals.real)[::2])

        # Base thermal states (nu_k = 0.5 -> vacuum).
        rho_list = []
        for nu_k in nu:
            if nu_k < TOL_SYMPLECTIC_FLOOR:
                nu_k = 0.5
            n_thermal = nu_k - 0.5
            if n_thermal < 1e-6:
                rho_list.append(qt.ket2dm(qt.fock(N_cutoff, 0)))
            else:
                rho_list.append(qt.thermal_dm(N_cutoff, n_thermal))
        rho_0 = qt.tensor(*rho_list)

        # Canonical operators in the combined Hilbert space.
        a_ops = []
        for i in range(n_modes):
            op_list = [qt.qeye(N_cutoff)] * n_modes
            op_list[i] = qt.destroy(N_cutoff)
            a_ops.append(qt.tensor(*op_list))

        r_ops = []
        for a in a_ops:
            x = (a + a.dag()) / np.sqrt(2)
            p = (a - a.dag()) / (1j * np.sqrt(2))
            r_ops.extend([x, p])

        # Symplectic transform S with S @ V_diag @ S.T == V, via matrix square roots.
        V_diag = scipy.linalg.block_diag(*[nu_k * np.eye(2) for nu_k in nu])
        S = scipy.linalg.sqrtm(self.covariance @ np.linalg.inv(V_diag)).real

        # Quadratic generator: S = exp(Omega @ G) -> G = -Omega @ logm(S).
        G = -Omega @ scipy.linalg.logm(S).real

        H_cv = 0
        for i in range(2 * n_modes):
            for j in range(2 * n_modes):
                if np.abs(G[i, j]) > TOL_ZERO_ENTRY:
                    H_cv += 0.5 * G[i, j] * r_ops[i] * r_ops[j]

        U_cv = (-1j * H_cv).expm()
        rho_transformed = U_cv * rho_0 * U_cv.dag()

        # Displacement.
        H_disp = 0
        for i in range(n_modes):
            d_x = self.displacement[2 * i]
            d_p = self.displacement[2 * i + 1]
            if np.abs(d_x) > TOL_ZERO_ENTRY or np.abs(d_p) > TOL_ZERO_ENTRY:
                H_disp += d_p * r_ops[2 * i] - d_x * r_ops[2 * i + 1]

        if H_disp != 0:
            D_cv = (-1j * H_disp).expm()
            rho_final = D_cv * rho_transformed * D_cv.dag()
        else:
            rho_final = rho_transformed

        trace_err = abs(rho_final.tr() - 1.0)
        if trace_err > TOL_TRACE_WARN:
            logger.warning(
                "to_qutip: tr(rho) deviates from 1 by %.2e at N_cutoff=%d. Try a "
                "larger N_cutoff first (Fock truncation); if the deviation doesn't "
                "shrink, it's sqrtm/logm precision loss in the symplectic "
                "decomposition for this covariance matrix, not truncation.",
                trace_err,
                N_cutoff,
            )

        return rho_final

    # -- Plotting -------------------------------------------------------

    def plot_covariance(self):
        """Visualize correlations between all registered modes."""
        plt = _lazy_pyplot()
        ticks = []
        for m in self.modes:
            ticks.extend([f"q_{m}", f"p_{m}"])

        plt.figure(figsize=(6, 5))
        im = plt.imshow(self.covariance, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Variance / covariance")
        plt.xticks(range(len(ticks)), ticks)
        plt.yticks(range(len(ticks)), ticks)
        plt.title("Multi-mode covariance matrix V")
        plt.show()

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
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> GaussianState:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Gaussian unitary operations (gates)
# ---------------------------------------------------------------------------


class GaussianOperations:
    @staticmethod
    def create_vacuum(modes: tuple[str, ...]) -> GaussianState:
        """Multi-mode vacuum state (V = 0.5 * I)."""
        dim = 2 * len(modes)
        d = np.zeros(dim)
        V = 0.5 * np.eye(dim)
        return GaussianState(modes=modes, displacement=d, covariance=V)

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

        new_d = S_global @ state.displacement
        new_V = S_global @ state.covariance @ S_global.T
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)

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

        new_d = S_BS @ state.displacement
        new_V = S_BS @ state.covariance @ S_BS.T
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)

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

        new_d = X @ state.displacement
        new_V = X @ state.covariance @ X.T + Y
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)


# ---------------------------------------------------------------------------
# General Gaussian channels
# ---------------------------------------------------------------------------


@dataclass
class GaussianChannel:
    """A general Gaussian channel d' = X@d + d0, V' = X@V@X.T + Y acting on
    a subset of modes."""

    target_modes: tuple[str, ...]
    X: np.ndarray
    Y: np.ndarray
    d0: np.ndarray

    def __post_init__(self):
        dim = 2 * len(self.target_modes)
        if self.X.shape != (dim, dim):
            raise ValueError(f"X must have shape ({dim}, {dim}), got {self.X.shape}.")
        if self.Y.shape != (dim, dim):
            raise ValueError(f"Y must have shape ({dim}, {dim}), got {self.Y.shape}.")
        if self.d0.shape != (dim,):
            raise ValueError(f"d0 must have shape ({dim},), got {self.d0.shape}.")

    def apply(self, state: GaussianState) -> GaussianState:
        global_dim = len(state.displacement)

        X_global = np.eye(global_dim)
        Y_global = np.zeros((global_dim, global_dim))
        d0_global = np.zeros(global_dim)

        for local_idx1, m1 in enumerate(self.target_modes):
            gi1 = state.get_mode_index(m1)
            d0_global[gi1 : gi1 + 2] = self.d0[local_idx1 * 2 : local_idx1 * 2 + 2]

            for local_idx2, m2 in enumerate(self.target_modes):
                gi2 = state.get_mode_index(m2)
                X_global[gi1 : gi1 + 2, gi2 : gi2 + 2] = self.X[
                    local_idx1 * 2 : local_idx1 * 2 + 2,
                    local_idx2 * 2 : local_idx2 * 2 + 2,
                ]
                Y_global[gi1 : gi1 + 2, gi2 : gi2 + 2] = self.Y[
                    local_idx1 * 2 : local_idx1 * 2 + 2,
                    local_idx2 * 2 : local_idx2 * 2 + 2,
                ]

        new_d = X_global @ state.displacement + d0_global
        new_V = X_global @ state.covariance @ X_global.T + Y_global
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)


class QBSChannels:
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
        Y = np.diag([0.0, sigma_phi**2])
        d0 = np.zeros(2)
        return GaussianChannel(target_modes=(mode,), X=X, Y=Y, d0=d0)

    @staticmethod
    def correlated_thermal_noise(
        mode_a: str, mode_b: str, eta: float, n_thermal: float, c_correlation: float
    ) -> GaussianChannel:
        """Correlated thermal noise on two modes coupled to the same bath."""
        _check_unit_interval(eta, "eta")
        _check_non_negative(n_thermal, "n_thermal")

        X = np.sqrt(eta) * np.eye(4)
        V_diag = (1 - eta) * (n_thermal + 0.5) * np.eye(2)
        V_cross = (1 - eta) * c_correlation * np.eye(2)
        Y = np.block([[V_diag, V_cross], [V_cross.T, V_diag]])
        d0 = np.zeros(4)
        return GaussianChannel(target_modes=(mode_a, mode_b), X=X, Y=Y, d0=d0)


# ---------------------------------------------------------------------------
# Circuit compiler
# ---------------------------------------------------------------------------


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
    return QBSChannels.thermal_loss(mode=modes[0], **kwargs).apply(state)


# Registry mapping a CircuitOperation.name -> (state, modes, **kwargs) -> state.
# Extend the circuit vocabulary by adding entries here (or via
# GaussianCircuit.register) instead of touching compile_and_run.
OPERATION_REGISTRY: dict[str, Callable[..., GaussianState]] = {
    "Squeezing": _op_squeeze,
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

    @classmethod
    def register(cls, name: str, fn: Callable[..., GaussianState]) -> None:
        """Register a new circuit-operation kind so `.compile_and_run` can execute it."""
        OPERATION_REGISTRY[name] = fn

    def add_mode(self, mode_name: str) -> GaussianCircuit:
        if mode_name in self.modes:
            raise ValueError(
                f"Mode '{mode_name}' is already registered in this circuit."
            )
        self.modes = self.modes + (mode_name,)
        return self

    def _add_op(self, name: str, modes: tuple[str, ...], **kwargs) -> GaussianCircuit:
        self._operations.append(CircuitOperation(name=name, modes=modes, kwargs=kwargs))
        return self

    def squeeze(self, mode: str, r: float, theta: float = 0.0) -> GaussianCircuit:
        return self._add_op("Squeezing", (mode,), r=r, theta=theta)

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
            current_state = GaussianOperations.create_vacuum(self.modes)
        else:
            if set(initial_state.modes) != set(self.modes):
                raise ValueError(
                    "Initial state's modes don't match the circuit's modes."
                )
            current_state = initial_state

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
            "operations": [
                {"name": op.name, "modes": list(op.modes), "kwargs": op.kwargs}
                for op in self._operations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianCircuit:
        circuit = cls(modes=tuple(data["modes"]))
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
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> GaussianCircuit:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


class GaussianMeasurements:
    @staticmethod
    def homodyne_measurement(
        state: GaussianState,
        measured_mode: str,
        phi: float,
        outcome: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, GaussianState]:
        """Stochastically-correct homodyne projection on `measured_mode`.
        Computes the phase-space collapse of the remaining modes via the
        Schur complement.

        Parameters
        ----------
        phi : local-oscillator phase (0 measures x/q, pi/2 measures p).
        outcome : if given, force this measurement result instead of sampling.
        rng : numpy Generator for reproducible sampling; defaults to a fresh
            `np.random.default_rng()` per call.
        """
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

        V_MM = V_rot[idx_x, idx_x]
        V_MR = V_rot[idx_x, remaining_indices]
        V_RM = V_rot[remaining_indices, idx_x]
        V_RR = V_rot[np.ix_(remaining_indices, remaining_indices)]

        d_M = d_rot[idx_x]
        d_R = d_rot[remaining_indices]

        if outcome is None:
            rng = rng if rng is not None else np.random.default_rng()
            measured_value = rng.normal(loc=d_M, scale=np.sqrt(V_MM))
        else:
            measured_value = outcome

        d_cond = d_R + V_RM * (1.0 / V_MM) * (measured_value - d_M)
        V_cond = V_RR - np.outer(V_RM, V_MR) / V_MM

        remaining_modes = tuple(m for m in state.modes if m != measured_mode)
        return measured_value, GaussianState(remaining_modes, d_cond, V_cond)


# ---------------------------------------------------------------------------
# Analytic phase-space plotting (no Hilbert space needed)
# ---------------------------------------------------------------------------


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
    plt = _lazy_pyplot()
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
    state: GaussianState, mode_a: str, mode_b: str, x_max: float = 3.0
):
    """Joint probability distribution of x_a vs x_b (e.g. EPR correlation)."""
    idx_a = state.get_mode_index(mode_a)
    idx_b = state.get_mode_index(mode_b)

    V_sub = np.array(
        [
            [state.covariance[idx_a, idx_a], state.covariance[idx_a, idx_b]],
            [state.covariance[idx_b, idx_a], state.covariance[idx_b, idx_b]],
        ]
    )
    d_sub = np.array([state.displacement[idx_a], state.displacement[idx_b]])

    xvec = np.linspace(-x_max, x_max, 150)
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


def plot_joint_correlation(P, X_a, X_b, mode_a: str, mode_b: str):
    plt = _lazy_pyplot()
    plt.figure(figsize=(6, 5))
    plt.contourf(X_a, X_b, P, 100, cmap="viridis")
    plt.colorbar(label="Probability density")
    plt.title(f"Correlation: quadrature x_{mode_a} vs x_{mode_b}")
    plt.xlabel(f"x_{mode_a}")
    plt.ylabel(f"x_{mode_b}")
    plt.axis("equal")
    plt.show()


# ---------------------------------------------------------------------------
# Fock-space (non-Gaussian) operations — single shared implementation
# ---------------------------------------------------------------------------


class FockOperations:
    """Low-level Fock-space photon click operations, shared by
    NonGaussianOperations (state-in/rho-out) and QBSSimulator (rho-in/rho-out)
    so the math lives in exactly one place."""

    @staticmethod
    def _mode_operator(op_1mode, n_modes: int, mode_idx: int, N_cutoff: int):
        qt = _lazy_qutip()
        if n_modes == 1:
            return op_1mode
        op_list = [qt.qeye(N_cutoff)] * n_modes
        op_list[mode_idx] = op_1mode
        return qt.tensor(*op_list)

    @staticmethod
    def _apply_and_renormalize(rho, op, label: str):
        rho_new = op * rho * op.dag()
        trace_val = rho_new.tr()
        if abs(trace_val) < TOL_ZERO_TRACE:
            raise ValueError(
                f"{label}: heralding success probability is numerically zero."
            )
        return rho_new / trace_val

    @staticmethod
    def photon_subtraction(rho, mode_idx: int = 0, N_cutoff: int = 20):
        """rho -> a * rho * a^dagger (renormalized). Probabilistic heralding."""
        qt = _lazy_qutip()
        n_modes = len(rho.dims[0])
        a_op = FockOperations._mode_operator(
            qt.destroy(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockOperations._apply_and_renormalize(rho, a_op, "photon_subtraction")

    @staticmethod
    def photon_addition(rho, mode_idx: int = 0, N_cutoff: int = 20):
        """rho -> a^dagger * rho * a (renormalized)."""
        qt = _lazy_qutip()
        n_modes = len(rho.dims[0])
        adag_op = FockOperations._mode_operator(
            qt.create(N_cutoff), n_modes, mode_idx, N_cutoff
        )
        return FockOperations._apply_and_renormalize(rho, adag_op, "photon_addition")


class NonGaussianOperations:
    """Convenience wrappers that take a GaussianState (converting to Fock
    space internally) instead of an already-converted qutip Qobj."""

    @staticmethod
    def photon_subtraction(state: GaussianState, mode_name: str, N_cutoff: int = 20):
        rho = state.to_qutip(N_cutoff=N_cutoff)
        mode_idx = state.modes.index(mode_name)
        return FockOperations.photon_subtraction(
            rho, mode_idx=mode_idx, N_cutoff=N_cutoff
        )

    @staticmethod
    def photon_addition(state: GaussianState, mode_name: str, N_cutoff: int = 20):
        rho = state.to_qutip(N_cutoff=N_cutoff)
        mode_idx = state.modes.index(mode_name)
        return FockOperations.photon_addition(rho, mode_idx=mode_idx, N_cutoff=N_cutoff)


# ---------------------------------------------------------------------------
# Time-dependent / interferometric simulation
# ---------------------------------------------------------------------------


class QBSSimulator:
    """Time-dependent master-equation and interferometer simulations."""

    # Backward/forward-compatible entry points onto the single FockOperations
    # implementation (useful when you already hold a converted rho, e.g. mid-
    # simulation, and have no GaussianState to go back to).
    photon_subtraction = staticmethod(FockOperations.photon_subtraction)
    photon_addition = staticmethod(FockOperations.photon_addition)

    @staticmethod
    def run_cavity_with_pulse(
        rho_init,
        tlist: np.ndarray,
        K: float,
        kappa: float,
        amp: float,
        t0: float,
        sigma: float,
        N_cutoff: int,
    ) -> list:
        """Dissipative cavity with a Kerr nonlinearity, driven by a
        time-dependent Gaussian pulse.

        Parameters
        ----------
        rho_init : starting density matrix (e.g. from GaussianState.to_qutip())
        tlist : time grid for the ODE solver
        K : Kerr nonlinearity strength (K * adag^2 * a^2)
        kappa : cavity photon-loss rate
        amp, t0, sigma : Gaussian pulse-shape parameters
        N_cutoff : Hilbert-space dimension
        """
        qt = _lazy_qutip()
        _check_non_negative(kappa, "kappa")

        a = qt.destroy(N_cutoff)
        H_kerr = K * a.dag() * a.dag() * a * a

        def pulse_shape(t, amp, t0, sigma):
            return amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

        H_total = [H_kerr, [a + a.dag(), pulse_shape]]
        c_ops = [np.sqrt(kappa) * a] if kappa > 0 else []
        args = {"amp": amp, "t0": t0, "sigma": sigma}

        res = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args=args)
        return res.states

    @staticmethod
    def scan_mzi_with_loss(
        psi_cat_single, theta_list: np.ndarray, kappa: float, N_cutoff: int
    ) -> dict:
        """Noisy Mach-Zehnder interferometer scan for a cat state: intensities
        and output parity as a function of the phase theta."""
        qt = _lazy_qutip()
        _check_non_negative(kappa, "kappa")

        a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
        a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

        n1_op = a1.dag() * a1
        n2_op = a2.dag() * a2
        parity1_op = (1j * np.pi * n1_op).expm()

        U_BS = ((1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())).expm()

        psi_in = qt.tensor(psi_cat_single, qt.fock(N_cutoff, 0))
        psi_after_BS1 = U_BS * psi_in

        results = {"theta": theta_list, "n1": [], "n2": [], "parity1": []}
        c_ops = [np.sqrt(kappa) * a1] if kappa > 0 else []
        H_phase = a1.dag() * a1

        for theta in theta_list:
            t_span = [0, theta] if theta > 0 else [0, 1e-9]
            sim = qt.mesolve(H_phase, psi_after_BS1, t_span, c_ops=c_ops)
            rho_arm = sim.states[-1]
            # mesolve returns a ket (not a density matrix) when there are no
            # collapse operators (kappa == 0). Normalize to rho either way so
            # the recombination step below works for both the lossy and
            # loss-free branches.
            if rho_arm.isket:
                rho_arm = qt.ket2dm(rho_arm)
            rho_out = U_BS * rho_arm * U_BS.dag()

            results["n1"].append(qt.expect(n1_op, rho_out))
            results["n2"].append(qt.expect(n2_op, rho_out))
            results["parity1"].append(qt.expect(parity1_op, rho_out).real)

        return results

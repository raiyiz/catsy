"""Gaussian-state simulation and analysis.

This module contains the continuous-variable Gaussian layer: states and
standard operations, general Gaussian channels, circuits, measurements, and
phase-space diagnostics. QuTiP is a required core dependency of catsy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.linalg

from .core import (
    TOL_PHYSICALITY,
    _apply_gaussian_transform,
    _check_non_negative,
    _check_positive_int,
    _check_thermal_correlation,
    _check_unit_interval,
    _json_load,
    _json_save,
    _symplectic_form,
    _validate_finite_array,
    _validate_gaussian_channel,
    _validate_physical_covariance,
    _williamson_decomposition,
)
from .types import (
    FloatArray,
    GaussianChannelData,
    GaussianCircuitData,
    GaussianStateData,
    Modes,
    OperationParameters,
    ParameterValue,
)

logger = logging.getLogger("catsy")


# ========================================================================
# Gaussian
# ========================================================================


def _qutip_passive_unitary(O: np.ndarray, a_ops: list[qt.Qobj]) -> qt.Qobj:
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

    # Starts as a plain int and, once any term is added below, becomes a
    # QuTiP Qobj -- qutip ships no type stubs, so its true dynamic type is
    # opaque to mypy regardless.
    H: qt.Qobj | None = None
    for i in range(n_modes):
        for j in range(n_modes):
            hij = h[i, j]
            if abs(hij) > TOL_PHYSICALITY:
                term = hij * a_ops[i].dag() * a_ops[j]
                H = term if H is None else H + term

    if H is None:
        return qt.tensor(*[qt.qeye(a.dims[0][0]) for a in a_ops])
    return (-1j * H).expm()


@dataclass
class GaussianState:
    """A multi-mode Gaussian state, fully described by (modes, d, V)."""

    modes: Modes
    displacement: FloatArray
    covariance: FloatArray

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        n_modes = len(self.modes)
        if len(set(self.modes)) != n_modes:
            raise ValueError(f"Duplicate mode names in {self.modes!r}.")
        expected_dim = 2 * n_modes
        if self.displacement.shape != (expected_dim,):
            raise ValueError(
                f"displacement must have shape ({expected_dim},), got {self.displacement.shape}."
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

    def reorder_modes(self, modes: Sequence[str]) -> GaussianState:
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

        if requested == self.modes:
            return self.copy()

        indices = [
            self.get_mode_index(mode) + offset for mode in requested for offset in (0, 1)
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

    # -- Constructors -------------------------------------------------------

    @classmethod
    def vacuum(cls, modes: Modes) -> GaussianState:
        """Return a multi-mode vacuum state (V = 0.5 * I)."""
        dim = 2 * len(modes)
        return cls(
            modes=modes,
            displacement=np.zeros(dim),
            covariance=0.5 * np.eye(dim),
        )

    @classmethod
    def coherent(cls, modes: Modes, alphas: complex | Sequence[complex]) -> GaussianState:
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

    @classmethod
    def tmsv(cls, mode_a: str, mode_b: str, r: float) -> GaussianState:
        """Return a two-mode squeezed vacuum (TMSV) state."""
        _check_non_negative(r, "r")
        return (
            cls.vacuum((mode_a, mode_b))
            .squeeze(mode_a, r=r, theta=0.0)
            .squeeze(mode_b, r=r, theta=np.pi / 2)
            .beam_splitter(mode_a, mode_b, eta=0.5)
        )

    # -- Gaussian transformations -----------------------------------------

    def squeeze(self, mode: str, r: float, theta: float = 0.0) -> GaussianState:
        """Return the state after single-mode squeezing on ``mode``."""
        idx = self.get_mode_index(mode)
        dim = len(self.displacement)
        S_local = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S_local = R @ S_local @ R.T
        S_global = np.eye(dim)
        S_global[idx : idx + 2, idx : idx + 2] = S_local
        return _apply_gaussian_transform(self, S_global)

    def rotate(self, mode: str, phi: float) -> GaussianState:
        """Return the state after phase-space rotation on ``mode``."""
        idx = self.get_mode_index(mode)
        dim = len(self.displacement)
        R_local = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        R_global = np.eye(dim)
        R_global[idx : idx + 2, idx : idx + 2] = R_local
        return _apply_gaussian_transform(self, R_global)

    def displace(
        self,
        mode: str,
        alpha: complex | None = None,
        *,
        x: float | None = None,
        p: float | None = None,
    ) -> GaussianState:
        """Return the state after displacing ``mode``.

        Give either ``alpha`` or both ``x`` and ``p``; supplying both forms is
        rejected.
        """
        if alpha is not None and (x is not None or p is not None):
            raise ValueError("Pass either `alpha` or (`x`, `p`), not both.")
        if alpha is not None:
            d_x, d_p = np.sqrt(2.0) * np.real(alpha), np.sqrt(2.0) * np.imag(alpha)
        elif x is not None and p is not None:
            d_x, d_p = x, p
        else:
            raise ValueError("Must supply either `alpha` or both `x` and `p`.")
        idx = self.get_mode_index(mode)
        new_d = self.displacement.copy()
        new_d[idx] += d_x
        new_d[idx + 1] += d_p
        return GaussianState(
            modes=self.modes, displacement=new_d, covariance=self.covariance.copy()
        )

    def beam_splitter(self, mode_a: str, mode_b: str, eta: float) -> GaussianState:
        """Return the state after a lossless beam splitter."""
        if mode_a == mode_b:
            raise ValueError("mode_a and mode_b must be different modes.")
        _check_unit_interval(eta, "eta")
        idx_a = self.get_mode_index(mode_a)
        idx_b = self.get_mode_index(mode_b)
        dim = len(self.displacement)
        t = np.sqrt(eta)
        r_coeff = np.sqrt(1 - eta)
        S_BS = np.eye(dim)
        I2 = np.eye(2)
        S_BS[idx_a : idx_a + 2, idx_a : idx_a + 2] = t * I2
        S_BS[idx_a : idx_a + 2, idx_b : idx_b + 2] = r_coeff * I2
        S_BS[idx_b : idx_b + 2, idx_a : idx_a + 2] = -r_coeff * I2
        S_BS[idx_b : idx_b + 2, idx_b : idx_b + 2] = t * I2
        return _apply_gaussian_transform(self, S_BS)

    def loss(self, mode: str, eta: float) -> GaussianState:
        """Return the state after vacuum-coupled loss on ``mode``."""
        _check_unit_interval(eta, "eta")
        idx = self.get_mode_index(mode)
        dim = len(self.displacement)
        X = np.eye(dim)
        X[idx : idx + 2, idx : idx + 2] = np.sqrt(eta) * np.eye(2)
        Y = np.zeros((dim, dim))
        Y[idx : idx + 2, idx : idx + 2] = (1 - eta) * 0.5 * np.eye(2)
        return _apply_gaussian_transform(self, X, noise=Y)

    # -- Fock-space bridge --------------------------------------------------

    def to_qutip(self, N_cutoff: int = 15) -> qt.Qobj:
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
            qt.thermal_dm(N_cutoff, max(nu - 0.5, 0.0)) for nu in symplectic_values
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

        H_positive: qt.Qobj | None = None
        for i in range(2 * n_modes):
            for j in range(2 * n_modes):
                gij = G[i, j]
                if np.abs(gij) > TOL_PHYSICALITY:
                    term = 0.5 * gij * r_ops[i] * r_ops[j]
                    H_positive = term if H_positive is None else H_positive + term

        if H_positive is not None:
            U_positive = (-1j * H_positive).expm()
            rho = U_positive * rho * U_positive.dag()

        # QuTiP's displacement convention is alpha=(dx+i*dp)/sqrt(2), which
        # directly matches the x,p convention used by GaussianState.
        for i in range(n_modes):
            dx = self.displacement[2 * i]
            dp = self.displacement[2 * i + 1]
            if abs(dx) <= TOL_PHYSICALITY and abs(dp) <= TOL_PHYSICALITY:
                continue

            alpha = (dx + 1j * dp) / np.sqrt(2.0)
            op_list = [qt.qeye(N_cutoff) for _ in range(n_modes)]
            op_list[i] = qt.displace(N_cutoff, alpha)
            D_op = qt.tensor(*op_list)
            rho = D_op * rho * D_op.dag()

        return rho

    # -- Plotting -------------------------------------------------------

    def plot_covariance(self) -> None:
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
        plt.show()

    # -- Serialization ----------------------------------------------------

    def to_dict(self) -> GaussianStateData:
        return {
            "modes": list(self.modes),
            "displacement": self.displacement.tolist(),
            "covariance": self.covariance.tolist(),
        }

    @classmethod
    def from_dict(cls, data: GaussianStateData) -> GaussianState:
        return cls(
            modes=tuple(data["modes"]),
            displacement=np.array(data["displacement"], dtype=float),
            covariance=np.array(data["covariance"], dtype=float),
        )

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GaussianState:
        return cls.from_dict(cast(GaussianStateData, _json_load(path)))


# ========================================================================
# Channels
# ========================================================================


@dataclass
class GaussianChannel:
    """A general Gaussian channel d' = X@d + d0, V' = X@V@X.T + Y acting on
    a subset of modes."""

    target_modes: Modes
    X: FloatArray
    Y: FloatArray
    d0: FloatArray

    def __post_init__(self) -> None:
        if len(set(self.target_modes)) != len(self.target_modes):
            raise ValueError(f"Duplicate target mode names in {self.target_modes!r}.")

        dim = 2 * len(self.target_modes)
        _validate_gaussian_channel(self.X, self.Y, self.d0, expected_dim=dim)

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

    def to_dict(self) -> GaussianChannelData:
        return {
            "target_modes": list(self.target_modes),
            "X": self.X.tolist(),
            "Y": self.Y.tolist(),
            "d0": self.d0.tolist(),
        }

    @classmethod
    def from_dict(cls, data: GaussianChannelData) -> GaussianChannel:
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
        return cls.from_dict(cast(GaussianChannelData, _json_load(path)))


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


# Gaussian operations are stored as the callable itself.  Serialization uses
# the function's bare ``__name__``; the small lookup below is only needed when
# loading serialized circuits and is never involved in execution.
class GaussianOperation(Protocol):
    __name__: str

    def __call__(
        self, state: GaussianState, modes: Modes, **kwargs: ParameterValue
    ) -> GaussianState: ...


# Built-in operation functions.  These are intentionally plain callables so a
# higher-level domain object can attach one directly to a circuit without an
# intermediate operation-data object or runtime name lookup.
def squeeze(
    state: GaussianState, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    return state.squeeze(
        mode=modes[0],
        r=cast(float, kwargs["r"]),
        theta=cast(float, kwargs["theta"]),
    )


def rotate(state: GaussianState, modes: Modes, **kwargs: ParameterValue) -> GaussianState:
    return state.rotate(mode=modes[0], phi=cast(float, kwargs["phi"]))


def displace(
    state: GaussianState, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    return state.displace(
        mode=modes[0],
        x=cast(float, kwargs["x"]),
        p=cast(float, kwargs["p"]),
    )


def beam_splitter(
    state: GaussianState, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    return state.beam_splitter(
        mode_a=modes[0],
        mode_b=modes[1],
        eta=cast(float, kwargs["eta"]),
    )


def loss(state: GaussianState, modes: Modes, **kwargs: ParameterValue) -> GaussianState:
    return state.loss(mode=modes[0], eta=cast(float, kwargs["eta"]))


def thermal_loss(
    state: GaussianState, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    channel = LossChannels.thermal_loss(
        mode=modes[0],
        eta=cast(float, kwargs["eta"]),
        n_thermal=cast(float, kwargs["n_thermal"]),
    )
    return channel.apply(state)


# This mapping is deliberately limited to deserialization.  Execution uses
# the callable stored in the circuit directly.
_OPERATION_DESERIALIZERS: dict[str, GaussianOperation] = {
    squeeze.__name__: squeeze,
    rotate.__name__: rotate,
    displace.__name__: displace,
    beam_splitter.__name__: beam_splitter,
    loss.__name__: loss,
    thermal_loss.__name__: thermal_loss,
}


@dataclass
class GaussianCircuit:
    """Sequences Gaussian operation callables and runs them over registered modes."""

    modes: Modes = field(default_factory=tuple)
    _operations: list[tuple[GaussianOperation, Modes, OperationParameters]] = field(
        default_factory=list, init=False
    )
    # Per-mode starting amplitude (0 -> vacuum), used by compile_and_run when
    # no initial_state is supplied -- see add_mode(alpha=...).
    _initial_alphas: dict[str, complex] = field(default_factory=dict, init=False)

    @classmethod
    def register(cls, name: str, fn: GaussianOperation) -> None:
        """Register a callable name for deserialization of custom operations.

        The callable itself is still what the circuit executes; this mapping is
        only consulted when reconstructing a circuit from serialized data.
        """
        _OPERATION_DESERIALIZERS[name] = fn

    def add_mode(self, mode_name: str, alpha: complex = 0.0) -> GaussianCircuit:
        """Register a new mode, optionally starting it in coherent state |alpha>."""
        if mode_name in self.modes:
            raise ValueError(f"Mode '{mode_name}' is already registered in this circuit.")
        self.modes = (*self.modes, mode_name)
        self._initial_alphas[mode_name] = alpha
        return self

    def add_operation(
        self,
        op: GaussianOperation,
        modes: Modes,
        **kwargs: ParameterValue,
    ) -> GaussianCircuit:
        """Attach an executable operation callable directly to the circuit."""
        normalized_modes = tuple(modes)
        if not normalized_modes:
            raise ValueError("A circuit operation must target at least one mode.")
        if any(
            not isinstance(mode, str) or not mode.strip() for mode in normalized_modes
        ):
            raise ValueError("All circuit operation modes must be non-empty strings.")
        if len(set(normalized_modes)) != len(normalized_modes):
            raise ValueError(
                f"{op.__name__} cannot target the same mode more than once: {normalized_modes!r}."
            )
        self._operations.append((op, normalized_modes, dict(kwargs)))
        return self

    def squeeze(self, mode: str, r: float, theta: float = 0.0) -> GaussianCircuit:
        return self.add_operation(squeeze, (mode,), r=r, theta=theta)

    def rotate(self, mode: str, phi: float) -> GaussianCircuit:
        return self.add_operation(rotate, (mode,), phi=phi)

    def displace(
        self,
        mode: str,
        alpha: complex | None = None,
        *,
        x: float | None = None,
        p: float | None = None,
    ) -> GaussianCircuit:
        """Add a displacement gate, storing it as real x/p parameters."""
        if alpha is not None and (x is not None or p is not None):
            raise ValueError("Pass either `alpha` or (`x`, `p`), not both.")
        if alpha is not None:
            x, p = np.sqrt(2.0) * np.real(alpha), np.sqrt(2.0) * np.imag(alpha)
        elif x is None or p is None:
            raise ValueError("Must supply either `alpha` or both `x` and `p`.")
        return self.add_operation(displace, (mode,), x=x, p=p)

    def beam_splitter(self, mode_a: str, mode_b: str, eta: float) -> GaussianCircuit:
        return self.add_operation(beam_splitter, (mode_a, mode_b), eta=eta)

    def loss(self, mode: str, eta: float) -> GaussianCircuit:
        return self.add_operation(loss, (mode,), eta=eta)

    def thermal_loss(self, mode: str, eta: float, n_thermal: float) -> GaussianCircuit:
        return self.add_operation(thermal_loss, (mode,), eta=eta, n_thermal=n_thermal)

    def compile_and_run(
        self, initial_state: GaussianState | None = None
    ) -> GaussianState:
        """Validate every operation against the registered modes and run the chain sequentially."""
        if not self.modes:
            raise ValueError("Circuit has no registered modes.")

        if initial_state is None:
            alphas = [self._initial_alphas.get(m, 0.0) for m in self.modes]
            current_state = GaussianState.coherent(self.modes, alphas)
        else:
            if set(initial_state.modes) != set(self.modes):
                raise ValueError("Initial state's modes don't match the circuit's modes.")
            current_state = initial_state.reorder_modes(self.modes)

        logger.debug(
            "Running circuit over modes %s (%d ops)", self.modes, len(self._operations)
        )

        for idx, (op, modes, kwargs) in enumerate(self._operations):
            for mode in modes:
                if mode not in self.modes:
                    raise ValueError(
                        f"Op #{idx} ({op.__name__}): mode '{mode}' is not registered in this circuit."
                    )
            current_state = op(current_state, modes, **kwargs)
            logger.debug(
                "[%d/%d] applied %s on %s",
                idx + 1,
                len(self._operations),
                op.__name__,
                modes,
            )

        return current_state

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> GaussianCircuitData:
        return {
            "modes": list(self.modes),
            "initial_alphas": {
                m: [float(np.real(a)), float(np.imag(a))]
                for m, a in self._initial_alphas.items()
            },
            "operations": [
                {"name": op.__name__, "modes": list(modes), "kwargs": kwargs}
                for op, modes, kwargs in self._operations
            ],
        }

    @classmethod
    def from_dict(cls, data: GaussianCircuitData) -> GaussianCircuit:
        circuit = cls(modes=tuple(data["modes"]))
        circuit._initial_alphas = {
            m: complex(re, im) for m, (re, im) in data.get("initial_alphas", {}).items()
        }
        for operation_data in data["operations"]:
            name = operation_data["name"]
            try:
                op = _OPERATION_DESERIALIZERS[name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown operation function '{name}' in serialized circuit."
                ) from exc
            circuit.add_operation(
                op, tuple(operation_data["modes"]), **operation_data["kwargs"]
            )
        return circuit

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> GaussianCircuit:
        return cls.from_dict(cast(GaussianCircuitData, _json_load(path)))


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
        if outcome is not None and (
            not isinstance(outcome, int | float) or not np.isfinite(outcome)
        ):
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

        if outcome is None:
            rng = rng if rng is not None else np.random.default_rng()
            measured_value = rng.normal(loc=d_M, scale=np.sqrt(V_MM))
        else:
            measured_value = outcome

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
        outcome: FloatArray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[FloatArray, GaussianState]:
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
        V_RR = state.covariance[np.ix_(remaining_indices, remaining_indices)]
        d_M = state.displacement[idx_m : idx_m + 2]
        d_R = state.displacement[remaining_indices]

        V_eff = 0.5 * (V_MM + 0.5 * np.eye(2)) + 0.5 * (V_MM + 0.5 * np.eye(2)).T
        if not np.all(np.isfinite(V_eff)):
            raise ValueError("heterodyne effective covariance must be finite.")

        try:
            np.linalg.cholesky(V_eff)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "heterodyne effective covariance must be positive definite."
            ) from exc

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
) -> tuple[FloatArray, FloatArray, FloatArray, str]:
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
    return W, X, P, mode_name


def plot_wigner(W: np.ndarray, X: np.ndarray, P: np.ndarray, mode_name: str) -> None:
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
) -> tuple[FloatArray, FloatArray, FloatArray, str, str]:
    """Joint probability distribution of the same quadrature on two modes
    (e.g. x_a vs x_b, or p_a vs p_b) -- the tool for actually *seeing* an
    EPR-style correlation or anti-correlation, as opposed to only reading it
    off the covariance matrix. `quadrature` selects which pair: 'x' shows
    position correlation, 'p' shows momentum correlation (anti-correlated,
    for the standard `GaussianState.tmsv` construction).
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
    return P, X_a, X_b, mode_a, mode_b


def plot_joint_correlation(
    P: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    mode_a: str,
    mode_b: str,
    quadrature: str = "x",
) -> None:
    plt.figure(figsize=(6, 5))
    plt.contourf(X_a, X_b, P, 100, cmap="viridis")
    plt.colorbar(label="Probability density")
    plt.title(f"Correlation: quadrature {quadrature}_{mode_a} vs {quadrature}_{mode_b}")
    plt.xlabel(f"{quadrature}_{mode_a}")
    plt.ylabel(f"{quadrature}_{mode_b}")
    plt.axis("equal")
    plt.show()


def compute_duan_inseparability(state: GaussianState, mode_a: str, mode_b: str) -> float:
    """Duan-Simon two-mode entanglement witness (Duan, Giedke, Cirac & Zoller,
    PRL 84, 2722 (2000)). In this codebase's vacuum=0.5 convention, *every*
    separable (classically-correlated-at-best) state satisfies

        Var(x_a - x_b) + Var(p_a + p_b) >= DUAN_SEPARABILITY_BOUND (== 2.0),

    with two independent vacua exactly saturating the bound. A value
    strictly below it is a *sufficient* condition for genuine, non-classical
    entanglement between mode_a and mode_b: no amount of classical
    correlation (e.g. from `LossChannels.correlated_thermal_noise`) can beat
    it, only a genuinely entangling operation like the beam splitter in
    `GaussianState.tmsv` can. The bound is not necessary --
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
    return float(var_x_diff + var_p_sum)

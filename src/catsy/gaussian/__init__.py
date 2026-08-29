"""Gaussian-state simulation and analysis.

This module contains the continuous-variable Gaussian layer: states and
standard operations, general Gaussian channels, circuits, measurements, and
phase-space diagnostics. QuTiP is a required core dependency of catsy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import qutip as qt
import scipy.linalg

from catsy.core import (
    TOL_PHYSICALITY,
    _apply_gaussian_transform,
    _check_non_negative,
    _check_positive_int,
    _check_thermal_correlation,
    _check_unit_interval,
    _json_load,
    _json_save,
    _normalize_phase_vector,
    _qutip_passive_unitary,
    _symplectic_form,
    _validate_finite_array,
    _validate_gaussian_channel,
    _validate_physical_covariance,
    _williamson_decomposition,
)
from catsy.fock import FockState
from catsy.types import (
    FloatArray,
    GaussianChannelData,
    GaussianStateData,
    Modes,
    ParameterValue,
)

logger = logging.getLogger("catsy")


# ========================================================================
# Gaussian
# ========================================================================


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
        _, d_x, d_p = _normalize_phase_vector(alpha=alpha, x=x, p=p)
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

    def to_fock(self, N_cutoff: int = 15) -> FockState:
        """Embed this Gaussian state into a truncated Fock-space representation.

        The conversion uses a numerically stable Williamson decomposition of
        the covariance matrix, followed by a polar decomposition of the
        resulting symplectic matrix.  The thermal Williamson modes are
        prepared with ``qutip.thermal_dm``; the passive part is implemented
        with a number-conserving quadratic Hamiltonian and the positive
        symplectic part with a quadratic quadrature Hamiltonian.  The
        displacement is applied with QuTiP's ``displace`` primitive.

        Williamson's decomposition is exact mathematically; this
        implementation verifies the reconstructed symplectic transformation
        and covariance to floating-point tolerance. The returned state is
        nevertheless represented in a finite Fock-space cutoff, so the final
        phase-space to Hilbert-space conversion can still incur truncation
        error.

        This is an *embedding*, not a reversible change of representation:
        every ``GaussianState`` has an exact (up to truncation) image in
        Fock space, but not every Fock state has a Gaussian phase-space
        description, so there is deliberately no ``FockState.to_gaussian()``
        counterpart -- see :mod:`catsy.fock`. Once a computation needs the
        resulting :class:`~catsy.fock.FockState`, it stays in that
        representation.
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

        return FockState(modes=self.modes, rho=rho, N_cutoff=N_cutoff)

    def to_qutip(self, N_cutoff: int = 15) -> qt.Qobj:
        """Deprecated alias for ``to_fock(N_cutoff).rho``.

        Kept for backward compatibility; new code should call
        :meth:`to_fock`, which additionally carries the state's mode names
        and is what `Circuit` uses to auto-promote into Fock space.
        """
        return self.to_fock(N_cutoff).rho

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
# Transforms
# ========================================================================

logger = logging.getLogger("catsy")


# Gaussian transforms remain ordinary functions. A :class:`Gate` binds one
# transform to its concrete name, target modes, and parameters.


def squeeze(
    state: GaussianState, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    return state.squeeze(
        mode=modes[0],
        r=cast(float, kwargs["r"]),
        theta=cast(float, kwargs.get("theta", 0.0)),
    )


def rotate(state: GaussianState, modes: Modes, **kwargs: ParameterValue) -> GaussianState:
    return state.rotate(mode=modes[0], phi=cast(float, kwargs["phi"]))


def displace(
    state: GaussianState, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    return state.displace(
        mode=modes[0],
        alpha=cast(complex, kwargs["alpha"]) if "alpha" in kwargs else None,
        x=cast(float, kwargs["x"]) if "x" in kwargs else None,
        p=cast(float, kwargs["p"]) if "p" in kwargs else None,
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


def initial_state(
    state: GaussianState | None, modes: Modes, **kwargs: ParameterValue
) -> GaussianState:
    """Construct the initial Gaussian state for a circuit."""
    if state is not None:
        return state
    kind = cast(str, kwargs.get("kind", "vacuum"))
    if kind == "vacuum":
        return GaussianState.vacuum(modes)
    if kind == "coherent":
        alpha, _, _ = _normalize_phase_vector(
            alpha=cast(complex, kwargs["alpha"]) if "alpha" in kwargs else None,
            x=cast(float, kwargs["x"]) if "x" in kwargs else None,
            p=cast(float, kwargs["p"]) if "p" in kwargs else None,
        )
        return GaussianState.coherent(modes, alpha)
    if kind == "tmsv":
        if len(modes) != 2:
            raise ValueError("tmsv initial state requires exactly two modes.")
        return GaussianState.tmsv(modes[0], modes[1], cast(float, kwargs["r"]))
    raise ValueError(f"Unknown Gaussian initial state kind {kind!r}.")


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

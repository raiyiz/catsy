"""Shared numerical helpers, conventions, and the generic executable circuit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import scipy.linalg

from .types import CircuitData, FloatArray, GateParameters, Modes, ParameterValue

if TYPE_CHECKING:
    from .gaussian import GaussianState


class GateTransform(Protocol):
    """Callable contract for a state transformation.

    Concrete transforms (``squeeze``, ``rotate``, ...) require a real
    ``GaussianState``. The one exception is ``InitialState``, whose own
    signature accepts ``GaussianState | None`` -- a function that accepts a
    wider input type than this protocol requires still satisfies it.
    """

    def __call__(
        self, state: GaussianState, modes: Modes, **kwargs: ParameterValue
    ) -> GaussianState: ...


@dataclass(frozen=True)
class Gate:
    """One fully bound transformation over named modes."""

    name: str
    transform: GateTransform
    modes: Modes
    kwargs: GateParameters

    def apply(self, state: Any | None) -> GaussianState:
        # `state` is None only when this is (or precedes) an InitialState
        # gate; Circuit.run() enforces that invariant before calling apply()
        # on any other gate, so the cast reflects an already-checked fact
        # rather than papering over an unchecked one.
        return self.transform(cast("GaussianState", state), self.modes, **self.kwargs)


class CircuitState(Protocol):
    """Minimal state interface required by :class:`Circuit`."""

    modes: Modes

    def reorder_modes(self, modes: Modes) -> Any: ...


@dataclass
class Circuit:
    """An ordered sequence of executable gates over named modes."""

    modes: Modes = field(default_factory=tuple)
    _gates: list[Gate] = field(default_factory=list, init=False)

    @classmethod
    def register(cls, name: str, transform: GateTransform) -> None:
        """Register a gate name and its transformation for construction/loading."""
        if not name.strip():
            raise ValueError("Gate name must be a non-empty string.")
        _GATE_DESERIALIZERS[name] = transform

    def __getattr__(self, name: str) -> Any:
        """Expose registered gates as fluent circuit-building methods.

        A resolved method appends the gate to this circuit; it never executes
        the gate against a state. This keeps the convenience API identical to
        ``add_gate`` while allowing ``circuit.squeeze(...)`` style DSLs.
        """
        transform = next(
            (
                candidate
                for candidate in _GATE_DESERIALIZERS.values()
                if getattr(candidate, "__name__", "") == name
            ),
            None,
        )
        if transform is None:
            raise AttributeError(name)
        gate_name = next(
            registered_name
            for registered_name, candidate in _GATE_DESERIALIZERS.items()
            if candidate is transform
        )

        def apply(*modes: str, **kwargs: ParameterValue) -> Circuit:
            return self.add_gate(
                Gate(
                    name=gate_name,
                    transform=transform,
                    modes=tuple(modes),
                    kwargs=dict(kwargs),
                )
            )

        return apply

    def add_mode(self, mode_name: str) -> Circuit:
        if mode_name in self.modes:
            raise ValueError(f"Mode '{mode_name}' is already registered in this circuit.")
        self.modes = (*self.modes, mode_name)
        return self

    def add_gate(self, gate: Gate) -> Circuit:
        normalized_modes = tuple(gate.modes)
        if not normalized_modes:
            raise ValueError("A circuit gate must target at least one mode.")
        if any(
            not isinstance(mode, str) or not mode.strip() for mode in normalized_modes
        ):
            raise ValueError("All circuit gate modes must be non-empty strings.")
        if len(set(normalized_modes)) != len(normalized_modes):
            raise ValueError(
                f"{gate.name} cannot target the same mode more than once: {normalized_modes!r}."
            )
        self._gates.append(gate)
        return self

    def initial_state(
        self, *modes: str, kind: str = "vacuum", **kwargs: ParameterValue
    ) -> Circuit:
        """Append the registered ``initial_state`` gate to the circuit."""
        try:
            transform = _GATE_DESERIALIZERS["InitialState"]
        except KeyError as exc:
            raise RuntimeError("No InitialState gate has been registered.") from exc
        return self.add_gate(
            Gate(
                name="InitialState",
                transform=transform,
                modes=tuple(modes),
                kwargs={"kind": kind, **kwargs},
            )
        )

    def run(self, initial_state: GaussianState | None = None) -> Any:
        """Run the gate chain, optionally constructing the state from a first gate.

        If ``initial_state`` is given, it is reordered to match this
        circuit's mode order and used as-is. If it is omitted, the circuit's
        own ``InitialState`` gate (added via :meth:`initial_state`) must be
        the first gate and is responsible for constructing the state; every
        other gate requires a state to already exist.
        """
        if not self.modes:
            raise ValueError("Circuit has no registered modes.")

        current_state: Any = None
        if initial_state is not None:
            if set(initial_state.modes) != set(self.modes):
                raise ValueError("Initial state's modes don't match the circuit's modes.")
            current_state = initial_state.reorder_modes(self.modes)

        for idx, gate in enumerate(self._gates):
            for mode in gate.modes:
                if mode not in self.modes:
                    raise ValueError(
                        f"Gate #{idx} ({gate.name}): mode '{mode}' is not registered in this circuit."
                    )
            if current_state is None and gate.name != "InitialState":
                raise ValueError(
                    f"Gate #{idx} ({gate.name}) cannot run before an initial_state gate."
                )
            current_state = gate.apply(current_state)

        if current_state is None:
            raise ValueError(
                "Circuit has to be initialized with a state: pass initial_state to "
                "run(), or add an InitialState gate via Circuit.initial_state(...)."
            )

        return current_state

    def to_dict(self) -> CircuitData:
        return {
            "modes": list(self.modes),
            "gates": [
                {"gate": gate.name, "modes": list(gate.modes), "kwargs": gate.kwargs}
                for gate in self._gates
            ],
        }

    @classmethod
    def from_dict(cls, data: CircuitData) -> Circuit:
        circuit = cls(modes=tuple(data["modes"]))
        for gate_data in data["gates"]:
            name = gate_data["gate"]
            try:
                transform = _GATE_DESERIALIZERS[name]
            except KeyError as exc:
                raise KeyError(f"Unknown gate '{name}' in serialized circuit.") from exc
            circuit.add_gate(
                Gate(
                    name=name,
                    transform=transform,
                    modes=tuple(gate_data["modes"]),
                    kwargs=dict(gate_data["kwargs"]),
                )
            )
        return circuit

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Circuit:
        return cls.from_dict(cast(CircuitData, _json_load(path)))


_GATE_DESERIALIZERS: dict[str, GateTransform] = {}

TOL_ZERO_ENTRY = 1e-9
TOL_TRACE_WARN = 1e-6
TOL_PHYSICALITY = 1e-10
DUAN_SEPARABILITY_BOUND = 2.0


def _check_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")


def _check_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value}.")


def _check_positive_int(value: int, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _symplectic_form(n_modes: int) -> np.ndarray:
    """Return Omega for the (x1, p1, x2, p2, ...) convention."""
    omega_1 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return scipy.linalg.block_diag(*[omega_1 for _ in range(n_modes)])


def _validate_finite_array(value: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_physical_covariance(
    covariance: np.ndarray, *, tol: float = TOL_PHYSICALITY
) -> None:
    """Validate V + i Omega / 2 >= 0 in the vacuum=0.5 convention."""
    if covariance.ndim != 2:
        raise ValueError("covariance must be a 2D array.")

    dim = covariance.shape[0]
    if dim % 2:
        raise ValueError(f"covariance dimension must be even, got {dim}.")

    _validate_finite_array(covariance, "covariance")
    if dim == 0:
        return

    if not np.allclose(covariance, covariance.T, atol=tol, rtol=0.0):
        raise ValueError("covariance must be symmetric.")

    omega = _symplectic_form(dim // 2)
    uncertainty_matrix = covariance + 0.5j * omega
    min_eigenvalue = np.linalg.eigvalsh(uncertainty_matrix).min()

    if min_eigenvalue < -tol:
        raise ValueError(
            "covariance violates the Gaussian uncertainty relation: "
            f"minimum eigenvalue of V + iOmega/2 is {min_eigenvalue:.3e}."
        )


def _validate_gaussian_channel(
    X: np.ndarray,
    Y: np.ndarray,
    d0: np.ndarray,
    *,
    expected_dim: int,
    tol: float = TOL_PHYSICALITY,
) -> None:
    """Validate dimensions and complete positivity of a Gaussian channel."""
    if X.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"X must have shape ({expected_dim}, {expected_dim}), got {X.shape}."
        )
    if Y.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Y must have shape ({expected_dim}, {expected_dim}), got {Y.shape}."
        )
    if d0.shape != (expected_dim,):
        raise ValueError(f"d0 must have shape ({expected_dim},), got {d0.shape}.")

    _validate_finite_array(X, "X")
    _validate_finite_array(Y, "Y")
    _validate_finite_array(d0, "d0")

    if not np.allclose(Y, Y.T, atol=tol, rtol=0.0):
        raise ValueError("Gaussian channel noise matrix Y must be symmetric.")

    omega = _symplectic_form(expected_dim // 2)
    cp_matrix = Y + 0.5j * (omega - X @ omega @ X.T)
    min_eigenvalue = np.linalg.eigvalsh(cp_matrix).min()

    if min_eigenvalue < -tol:
        raise ValueError(
            "Gaussian channel violates complete positivity: "
            "minimum eigenvalue of Y + i(Omega - XOmegaX^T)/2 is "
            f"{min_eigenvalue:.3e}."
        )


def _check_thermal_correlation(c_correlation: float, n_thermal: float) -> None:
    """Validate cross-mode thermal covariance for the shared-bath model."""
    if not np.isfinite(c_correlation):
        raise ValueError(f"c_correlation must be finite, got {c_correlation}.")

    # The underlying two-mode environment covariance is
    # [[(n+1/2)I, cI], [cI, (n+1/2)I]].
    # Its physicality condition is |c| <= n.
    limit = n_thermal
    if abs(c_correlation) > limit + TOL_PHYSICALITY:
        raise ValueError(
            "c_correlation is outside the physical range for the requested "
            f"thermal occupation: |c_correlation| must be <= {limit}, "
            f"got {c_correlation}."
        )


def _json_save(obj: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj))


def _json_load(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def _apply_gaussian_transform(
    state: GaussianState,
    transform: np.ndarray,
    noise: np.ndarray | None = None,
    displacement: np.ndarray | None = None,
) -> GaussianState:
    """Apply d' = S d + d0 and V' = S V Sᵀ + Y."""
    from .gaussian import GaussianState

    new_d = transform @ state.displacement
    if displacement is not None:
        new_d += displacement
    new_V = transform @ state.covariance @ transform.T
    if noise is not None:
        new_V += noise
    return GaussianState(state.modes, new_d, new_V)


def _williamson_decomposition(
    covariance: FloatArray,
    *,
    tol: float = 1e-10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return symplectic eigenvalues, S and D with V = S D S.T.

    The construction uses the positive square root of V followed by a real
    Schur decomposition of sqrt(V) @ Omega @ sqrt(V).  This avoids treating a
    generic matrix square root of V @ D^{-1} as though it were symplectic.
    """
    covariance = np.asarray(covariance, dtype=float)
    dim = covariance.shape[0]
    if dim == 0 or dim % 2:
        raise ValueError("covariance dimension must be a positive even number")

    n_modes = dim // 2
    Omega = _symplectic_form(n_modes)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    if np.min(eigvals) <= 0:
        raise ValueError("covariance must be positive definite")
    A = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T

    M = A @ Omega @ A
    T, O = scipy.linalg.schur(M, output="real")

    nus: list[float] = []
    for i in range(0, dim, 2):
        block = T[i : i + 2, i : i + 2]
        offdiag = 0.5 * (block[0, 1] - block[1, 0])
        nu = abs(offdiag)
        if nu <= tol:
            raise ValueError(
                "covariance has a numerically singular symplectic eigenvalue"
            )

        # Normalize the Schur block to +nu * [[0,1],[-1,0]].
        if offdiag < 0:
            O[:, i : i + 2] = O[:, i : i + 2] @ np.diag([1.0, -1.0])
        nus.append(nu)

    D_diag = np.repeat(nus, 2)
    D = np.diag(D_diag)
    S = A @ O @ np.diag(1.0 / np.sqrt(D_diag))

    symplectic_residual = np.max(np.abs(S @ Omega @ S.T - Omega))
    covariance_residual = np.max(np.abs(S @ D @ S.T - covariance))
    if symplectic_residual > 1e-8 or covariance_residual > 1e-8:
        raise RuntimeError(
            "Williamson decomposition residual too large: "
            f"symplectic={symplectic_residual:.3e}, "
            f"covariance={covariance_residual:.3e}."
        )

    return np.asarray(nus, dtype=float), S, D

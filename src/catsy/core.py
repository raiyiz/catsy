"""Shared numerical helpers, conventions, and the generic executable circuit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import scipy.linalg

from .modes import Mode
from .types import CircuitData, FloatArray, GateParameters, ModeName, ParameterValue, RuntimeModes

if TYPE_CHECKING:
    from .gaussian import GaussianState


class GateTransform(Protocol):
    def __call__(
        self, state: GaussianState, modes: tuple[ModeName, ...], **kwargs: ParameterValue
    ) -> GaussianState: ...


@dataclass(frozen=True)
class Gate:
    """One fully bound transformation over runtime mode objects."""

    name: str
    transform: GateTransform
    modes: RuntimeModes
    kwargs: GateParameters

    def apply(self, state: Any | None) -> GaussianState:
        return self.transform(
            cast("GaussianState", state),
            tuple(mode.name for mode in self.modes),
            **self.kwargs,
        )


class CircuitState(Protocol):
    modes: tuple[ModeName, ...]

    def reorder_modes(self, modes: tuple[ModeName, ...]) -> Any: ...


@dataclass
class Circuit:
    """An ordered executable circuit whose runtime modes have explicit ownership."""

    name: str = "Untitled Circuit"
    modes: RuntimeModes = field(default_factory=tuple)
    _gates: list[Gate] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        initial_modes = tuple(self.modes)
        self.modes = ()
        for mode in initial_modes:
            self.add_mode(mode)

    @property
    def mode_names(self) -> tuple[ModeName, ...]:
        """The serialized/name view of the circuit's runtime modes."""
        return tuple(mode.name for mode in self.modes)

    @property
    def gates(self) -> tuple[Gate, ...]:
        return tuple(self._gates)

    @classmethod
    def register(cls, name: str, transform: GateTransform) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Gate name must be a non-empty string.")
        _GATE_DESERIALIZERS[name] = transform

    def __getattr__(self, name: str) -> Any:
        transform = next(
            (candidate for candidate in _GATE_DESERIALIZERS.values()
             if getattr(candidate, "__name__", "") == name),
            None,
        )
        if transform is None:
            raise AttributeError(name)
        gate_name = next(
            registered_name for registered_name, candidate in _GATE_DESERIALIZERS.items()
            if candidate is transform
        )

        def apply(*modes: Mode | str, **kwargs: ParameterValue) -> Circuit:
            resolved = tuple(self._resolve_mode(mode) for mode in modes)
            return self.add_gate(Gate(gate_name, transform, resolved, dict(kwargs)))

        return apply

    def _resolve_mode(self, mode: Mode | str) -> Mode:
        if isinstance(mode, Mode):
            if mode.owner is not self:
                raise ValueError(
                    f"Mode '{mode.name}' is not owned by this circuit."
                )
            return mode
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("Mode must be a non-empty string or a Mode instance.")
        for registered in self.modes:
            if registered.name == mode:
                return registered
        raise ValueError(f"Mode '{mode}' is not registered in this circuit.")

    def add_mode(self, mode: Mode | str) -> Circuit:
        """Adopt a runtime mode, or construct one from a name."""
        if isinstance(mode, str):
            mode = Mode(mode)
        if not isinstance(mode, Mode):
            raise TypeError("mode must be a Mode or non-empty string")
        if any(existing.name == mode.name for existing in self.modes):
            raise ValueError(f"Mode '{mode.name}' is already registered in this circuit.")
        mode.adopt(self)
        self.modes = (*self.modes, mode)
        return self

    def remove_mode(self, mode: Mode | str) -> Circuit:
        """Release and remove a mode that no gate currently references."""
        target = self._resolve_mode(mode)
        if any(target in gate.modes for gate in self._gates):
            raise ValueError(f"Mode '{target.name}' is still referenced by a circuit gate.")
        self.modes = tuple(existing for existing in self.modes if existing is not target)
        target.release(self)
        return self

    def add_gate(self, gate: Gate) -> Circuit:
        raw_modes = tuple(gate.modes)
        if not raw_modes:
            raise ValueError("A circuit gate must target at least one mode.")
        resolved = tuple(self._resolve_mode(mode) for mode in raw_modes)
        if len({mode.name for mode in resolved}) != len(resolved):
            raise ValueError(
                f"{gate.name} cannot target the same mode more than once: "
                f"{tuple(mode.name for mode in resolved)!r}."
            )
        if resolved != raw_modes:
            gate = Gate(gate.name, gate.transform, resolved, gate.kwargs)
        self._gates.append(gate)
        return self

    def initial_state(
        self, *modes: Mode | str, kind: str = "vacuum", **kwargs: ParameterValue
    ) -> Circuit:
        try:
            transform = _GATE_DESERIALIZERS["InitialState"]
        except KeyError as exc:
            raise RuntimeError("No InitialState gate has been registered.") from exc
        return self.add_gate(
            Gate("InitialState", transform, tuple(self._resolve_mode(mode) for mode in modes),
                 {"kind": kind, **kwargs})
        )

    def run(self, initial_state: GaussianState | None = None) -> Any:
        if not self.modes:
            raise ValueError("Circuit has no registered modes.")

        current_state: Any = None
        names = self.mode_names
        if initial_state is not None:
            if set(initial_state.modes) != set(names):
                raise ValueError("Initial state's modes don't match the circuit's modes.")
            current_state = initial_state.reorder_modes(names)

        for idx, gate in enumerate(self._gates):
            if current_state is None and gate.name != "InitialState":
                raise ValueError(
                    f"Gate #{idx} ({gate.name}) cannot run before an initial_state gate."
                )
            current_state = gate.apply(current_state)

        if current_state is None:
            raise ValueError(
                "Circuit has to be initialized with a state: pass initial_state to run(), "
                "or add an InitialState gate via Circuit.initial_state(...)."
            )
        return current_state

    def to_dict(self) -> CircuitData:
        return {
            "name": self.name,
            "modes": list(self.mode_names),
            "gates": [
                {"gate": gate.name, "modes": [mode.name for mode in gate.modes], "kwargs": gate.kwargs}
                for gate in self._gates
            ],
        }

    @classmethod
    def from_dict(cls, data: CircuitData) -> Circuit:
        circuit = cls(name=data["name"], modes=tuple(data["modes"]))
        for gate_data in data["gates"]:
            name = gate_data["gate"]
            try:
                transform = _GATE_DESERIALIZERS[name]
            except KeyError as exc:
                raise KeyError(f"Unknown gate '{name}' in serialized circuit.") from exc
            circuit.add_gate(
                Gate(name, transform, tuple(circuit._resolve_mode(mode) for mode in gate_data["modes"]),
                     dict(gate_data["kwargs"]))
            )
        return circuit

    def save(self, path: str | Path) -> None:
        _json_save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Circuit:
        return cls.from_dict(cast(CircuitData, _json_load(path)))

    def render_schematic(self, input_states: dict[str, str] | None = None) -> str:
        if not self._gates:
            return f"┌─── {self.name} ───┐\n│   (Empty Circuit)   │\n└──────────────────────────┘"

        ordered_modes = list(self.mode_names)
        position = {mode: i for i, mode in enumerate(ordered_modes)}
        input_states = input_states or {}
        lines = {
            mode: f"{input_states.get(mode, '|0>'):<9} ──[{mode}]──"
            for mode in ordered_modes
        }
        for gate in self._gates:
            label, block_width = _render_gate_label(gate)
            involved_modes = sorted((mode.name for mode in gate.modes), key=position.__getitem__)
            first_pos = position[involved_modes[0]]
            last_pos = position[involved_modes[-1]]
            for mode in ordered_modes:
                pos = position[mode]
                if mode in involved_modes:
                    if len(involved_modes) > 1:
                        connector = "┬" if pos == first_pos else "┴" if pos == last_pos else "┼"
                        cell = f"[{connector}{label.center(block_width - 2)}{connector}]"
                    else:
                        cell = f"[{label.center(block_width)}]"
                    lines[mode] += f"─{cell}─"
                elif len(involved_modes) > 1 and first_pos < pos < last_pos:
                    bridge = "│".center(block_width + 2, "─")
                    lines[mode] += f"─{bridge}─"
                else:
                    lines[mode] += "─" * (block_width + 4)
        out = [f"┌{f'─┤ Schematic: {self.name} ├':─<78}┐"]
        out.extend(f"│  {lines[mode]}── OUT  │" for mode in ordered_modes)
        out.append("└" + "─" * 78 + "┘")
        return "\n".join(out)

    def draw(self, input_states: dict[str, str] | None = None) -> None:
        print("\n" + self.render_schematic(input_states) + "\n")


_GATE_DESERIALIZERS: dict[str, GateTransform] = {}
_GATE_LABEL_ABBREVIATIONS = {
    "BeamSplitter": "BS", "Noise": "LOSS", "Squeezer": "SQZ", "Rotator": "PHASE",
    "Displacer": "DISP", "ThermalLoss": "TLOSS", "InitialState": "INIT",
}


def _render_gate_label(gate: Gate) -> tuple[str, int]:
    param_str = ""
    for key, symbol in (("eta", "η"), ("phi", "φ"), ("r", "r"), ("alpha", "α")):
        if key in gate.kwargs:
            value = gate.kwargs[key]
            param_str = f" {symbol}={value:.2f}" if key == "phi" else f" {symbol}={value}"
            break
    type_name = _GATE_LABEL_ABBREVIATIONS.get(gate.name, gate.name[:5])
    label = f" {type_name}{param_str} "
    return label, max(len(label) + 2, 12)


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
    omega_1 = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return scipy.linalg.block_diag(*[omega_1 for _ in range(n_modes)])


def _validate_finite_array(value: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_physical_covariance(covariance: np.ndarray, *, tol: float = TOL_PHYSICALITY) -> None:
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
    min_eigenvalue = np.linalg.eigvalsh(covariance + 0.5j * omega).min()
    if min_eigenvalue < -tol:
        raise ValueError(
            "covariance violates the Gaussian uncertainty relation: "
            f"minimum eigenvalue of V + iOmega/2 is {min_eigenvalue:.3e}."
        )


def _validate_gaussian_channel(X: np.ndarray, Y: np.ndarray, d0: np.ndarray, *, expected_dim: int, tol: float = TOL_PHYSICALITY) -> None:
    if X.shape != (expected_dim, expected_dim):
        raise ValueError(f"X must have shape ({expected_dim}, {expected_dim}), got {X.shape}.")
    if Y.shape != (expected_dim, expected_dim):
        raise ValueError(f"Y must have shape ({expected_dim}, {expected_dim}), got {Y.shape}.")
    if d0.shape != (expected_dim,):
        raise ValueError(f"d0 must have shape ({expected_dim},), got {d0.shape}.")
    _validate_finite_array(X, "X")
    _validate_finite_array(Y, "Y")
    _validate_finite_array(d0, "d0")
    if not np.allclose(Y, Y.T, atol=tol, rtol=0.0):
        raise ValueError("Gaussian channel noise matrix Y must be symmetric.")
    omega = _symplectic_form(expected_dim // 2)
    min_eigenvalue = np.linalg.eigvalsh(Y + 0.5j * (omega - X @ omega @ X.T)).min()
    if min_eigenvalue < -tol:
        raise ValueError(
            "Gaussian channel violates complete positivity: minimum eigenvalue of "
            "Y + i(Omega - XOmegaX^T)/2 is "
            f"{min_eigenvalue:.3e}."
        )


def _check_thermal_correlation(c_correlation: float, n_thermal: float) -> None:
    if not np.isfinite(c_correlation):
        raise ValueError(f"c_correlation must be finite, got {c_correlation}.")
    if abs(c_correlation) > n_thermal + TOL_PHYSICALITY:
        raise ValueError(
            "c_correlation is outside the physical range for the requested "
            f"thermal occupation: |c_correlation| must be <= {n_thermal}, got {c_correlation}."
        )


def _json_save(obj: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj))


def _json_load(path: str | Path) -> object:
    return json.loads(Path(path).read_text())


def _apply_gaussian_transform(state: GaussianState, transform: np.ndarray, noise: np.ndarray | None = None, displacement: np.ndarray | None = None) -> GaussianState:
    from .gaussian import GaussianState
    new_d = transform @ state.displacement
    if displacement is not None:
        new_d += displacement
    new_V = transform @ state.covariance @ transform.T
    if noise is not None:
        new_V += noise
    return GaussianState(state.modes, new_d, new_V)


def _williamson_decomposition(covariance: FloatArray, *, tol: float = 1e-10) -> tuple[FloatArray, FloatArray, FloatArray]:
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
        block = T[i:i + 2, i:i + 2]
        offdiag = 0.5 * (block[0, 1] - block[1, 0])
        nu = abs(offdiag)
        if nu <= tol:
            raise ValueError("covariance has a numerically singular symplectic eigenvalue")
        if offdiag < 0:
            O[:, i:i + 2] = O[:, i:i + 2] @ np.diag([1.0, -1.0])
        nus.append(nu)
    D_diag = np.repeat(nus, 2)
    D = np.diag(D_diag)
    S = A @ O @ np.diag(1.0 / np.sqrt(D_diag))
    symplectic_residual = np.max(np.abs(S @ Omega @ S.T - Omega))
    covariance_residual = np.max(np.abs(S @ D @ S.T - covariance))
    if symplectic_residual > 1e-8 or covariance_residual > 1e-8:
        raise RuntimeError(
            "Williamson decomposition residual too large: "
            f"symplectic={symplectic_residual:.3e}, covariance={covariance_residual:.3e}."
        )
    return np.asarray(nus, dtype=float), S, D

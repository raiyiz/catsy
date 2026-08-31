"""Core circuit-building and execution primitives for catsy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from catsy.fock import FockState
from catsy.gaussian import GaussianState

from .core import _json_load, _json_save, _normalize_phase_vector
from .types import CircuitData, GateParameters, Modes, ParameterValue

CVState = GaussianState | FockState


class GateTransform(Protocol):
    """Callable contract for a state transformation."""

    def __call__(self, state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState: ...


@dataclass(frozen=True)
class Gate:
    """One fully bound transformation over named modes."""

    name: str
    transform: GateTransform
    modes: Modes
    kwargs: GateParameters

    def apply(self, state: Any | None) -> CVState:
        return self.transform(cast("CVState", state), self.modes, **self.kwargs)


class CircuitState(Protocol):
    """Minimal state interface required by :class:`Circuit`."""

    modes: Modes

    def reorder_modes(self, modes: Modes) -> Any: ...


@dataclass(frozen=True, eq=False, slots=True)
class Mode:
    """A named mode owned by an optional circuit."""

    name: str
    owner: Circuit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Mode name must be a non-empty string.")

    def __repr__(self) -> str:
        owner = f"owner={self.owner.name!r}" if self.owner is not None else "free"
        return f"Mode({self.name!r}, {owner})"


_GATE_DESERIALIZERS: dict[str, GateTransform] = {}
_GATE_LABEL_ABBREVIATIONS = {
    "BeamSplitter": "BS",
    "Noise": "LOSS",
    "Squeezer": "SQZ",
    "Rotator": "PHASE",
    "Displacer": "DISP",
    "ThermalLoss": "TLOSS",
    "InitialState": "INIT",
    "PhotonSubtraction": "PSUB",
    "PhotonAddition": "PADD",
    "RealisticPhotonSubtraction": "RPSUB",
    "RealisticPhotonAddition": "RPADD",
}


def _normalize_gate_kwargs(kwargs: dict[str, ParameterValue]) -> dict[str, ParameterValue]:
    """Make complex coherent amplitudes JSON-safe for serialized gates."""
    if "alpha" not in kwargs:
        return kwargs
    kwargs = dict(kwargs)
    _, x, p = _normalize_phase_vector(alpha=cast(complex, kwargs.pop("alpha")))
    kwargs["x"] = x
    kwargs["p"] = p
    return kwargs


@dataclass
class Circuit:
    """An ordered executable optical circuit over named modes."""

    name: str = "Untitled Circuit"
    modes: Modes = field(default_factory=tuple)
    _mode_registry: dict[str, Mode] = field(default_factory=dict, init=False)
    _gates: list[Gate] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        requested_modes, self.modes = self.modes, ()
        for mode_name in requested_modes:
            self.mode(mode_name)

    @property
    def gates(self) -> tuple[Gate, ...]:
        """The circuit's gates in execution order."""
        return tuple(self._gates)

    @classmethod
    def register(cls, name: str, transform: GateTransform) -> None:
        """Register a gate name for fluent construction and deserialization."""
        if not name.strip():
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
            registered for registered, candidate in _GATE_DESERIALIZERS.items()
            if candidate is transform
        )

        def apply(*modes: str | Mode, **kwargs: ParameterValue) -> Circuit:
            resolved = tuple(self._resolve_mode(mode) for mode in modes)
            return self.add_gate(
                Gate(gate_name, transform, resolved, _normalize_gate_kwargs(kwargs))
            )

        return apply

    def mode(self, mode_name: str) -> Mode:
        """Register a mode and return its circuit-owned handle."""
        if mode_name in self._mode_registry:
            raise ValueError(f"Mode '{mode_name}' is already registered in this circuit.")
        new_mode = Mode(mode_name, self)
        self._mode_registry[mode_name] = new_mode
        self.modes = (*self.modes, mode_name)
        return new_mode

    def add_mode(self, mode_name: str) -> Circuit:
        self.mode(mode_name)
        return self

    def _resolve_mode(self, mode: str | Mode) -> str:
        if isinstance(mode, Mode):
            if mode.owner is not self:
                owner_desc = "no circuit (a free/standalone mode)" if mode.owner is None else f"circuit {mode.owner.name!r}"
                raise ValueError(
                    f"Mode {mode.name!r} belongs to {owner_desc}, not to this circuit ({self.name!r})."
                )
            return mode.name
        if mode not in self._mode_registry:
            raise ValueError(
                f"Mode '{mode}' is not registered on this circuit. Call circuit.mode({mode!r}) first."
            )
        return mode

    def add_gate(self, gate: Gate) -> Circuit:
        modes = tuple(gate.modes)
        if not modes:
            raise ValueError("A circuit gate must target at least one mode.")
        if any(not isinstance(mode, str) or not mode.strip() for mode in modes):
            raise ValueError("All circuit gate modes must be non-empty strings.")
        if len(set(modes)) != len(modes):
            raise ValueError(f"{gate.name} cannot target the same mode more than once: {modes!r}.")
        unknown = [mode for mode in modes if mode not in self._mode_registry]
        if unknown:
            raise ValueError(
                f"{gate.name} targets mode(s) {unknown!r} not registered on this circuit ({self.name!r})."
            )
        self._gates.append(Gate(gate.name, gate.transform, modes, gate.kwargs))
        return self

    def initial_state(
        self, *modes: str | Mode, kind: str = "vacuum", **kwargs: ParameterValue
    ) -> Circuit:
        transform = _GATE_DESERIALIZERS.get("InitialState")
        if transform is None:
            raise RuntimeError("No InitialState gate has been registered.")
        resolved = tuple(self._resolve_mode(mode) for mode in modes)
        return self.add_gate(
            Gate("InitialState", transform, resolved, _normalize_gate_kwargs({"kind": kind, **kwargs}))
        )

    def run(self, initial_state: CVState | None = None) -> Any:
        if not self.modes:
            raise ValueError("Circuit has no registered modes.")
        current_state: Any = None
        if initial_state is not None:
            if set(initial_state.modes) != set(self.modes):
                raise ValueError("Initial state's modes don't match the circuit's modes.")
            current_state = initial_state.reorder_modes(self.modes)
        for idx, gate in enumerate(self._gates):
            if current_state is None and gate.name != "InitialState":
                raise ValueError(f"Gate #{idx} ({gate.name}) cannot run before an initial_state gate.")
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
            "modes": list(self.modes),
            "gates": [
                {"gate": gate.name, "modes": list(gate.modes), "kwargs": gate.kwargs}
                for gate in self._gates
            ],
        }

    @classmethod
    def from_dict(cls, data: CircuitData) -> Circuit:
        circuit = cls(name=data["name"], modes=tuple(data["modes"]))
        for gate_data in data["gates"]:
            try:
                transform = _GATE_DESERIALIZERS[gate_data["gate"]]
            except KeyError as exc:
                raise KeyError(f"Unknown gate '{gate_data['gate']}' in serialized circuit.") from exc
            circuit.add_gate(
                Gate(
                    name=gate_data["gate"],
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

    def render_schematic(self, input_states: dict[str, str] | None = None) -> str:
        """Render the circuit as a deterministic plain-text schematic."""
        if not self._gates:
            return f"┌─── {self.name} ───┐\n│   (Empty Circuit)   │\n└──────────────────────────┘"
        ordered = list(self.modes)
        position = {mode: i for i, mode in enumerate(ordered)}
        inputs = input_states or {}
        lines = {mode: f"{inputs.get(mode, '|0>'):<9} ──[{mode}]──" for mode in ordered}
        for gate in self._gates:
            label, width = _render_gate_label(gate)
            involved = sorted(gate.modes, key=position.__getitem__)
            first, last = position[involved[0]], position[involved[-1]]
            for mode in ordered:
                pos = position[mode]
                if mode in involved:
                    if len(involved) > 1:
                        connector = "┬" if pos == first else "┴" if pos == last else "┼"
                        cell = f"[{connector}{label.center(width - 2)}{connector}]"
                    else:
                        cell = f"[{label.center(width)}]"
                    lines[mode] += f"─{cell}─"
                elif len(involved) > 1 and first < pos < last:
                    lines[mode] += f"─{'│'.center(width + 2, '─')}─"
                else:
                    lines[mode] += "─" * (width + 4)
        title = f"─┤ Schematic: {self.name} ├"
        return "\n".join([f"┌{title:─<78}┐", *(f"│  {lines[m]}── OUT  │" for m in ordered), "└" + "─" * 78 + "┘"])

    def draw(self, input_states: dict[str, str] | None = None) -> None:
        print("\n" + self.render_schematic(input_states) + "\n")


def _render_gate_label(gate: Gate) -> tuple[str, int]:
    param_str = ""
    for key, symbol in (("eta", "η"), ("phi", "φ"), ("r", "r"), ("x", "x"), ("p", "p")):
        if key in gate.kwargs:
            value = gate.kwargs[key]
            param_str = f" {symbol}={value:.2f}" if isinstance(value, float) else f" {symbol}={value}"
            break
    type_name = _GATE_LABEL_ABBREVIATIONS.get(gate.name, gate.name[:5])
    label = f" {type_name}{param_str} "
    return label, max(len(label) + 2, 12)

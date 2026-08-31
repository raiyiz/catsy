"""Optical circuits and QuTiP-based physical simulations.

`Circuit` is the reusable optical-layout abstraction. The module also contains
physical, Fock-space models such as a driven Kerr cavity and a Mach-Zehnder
interferometer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import numpy as np
import qutip as qt

from catsy.fock import FockState, make_even_cat
from catsy.gaussian import GaussianState, initial_state
from catsy.gaussian import beam_splitter as _gaussian_beam_splitter
from catsy.gaussian import displace as _gaussian_displace
from catsy.gaussian import loss as _gaussian_loss
from catsy.gaussian import rotate as _gaussian_rotate
from catsy.gaussian import squeeze as _gaussian_squeeze
from catsy.gaussian import thermal_loss as _gaussian_thermal_loss

from .core import (
    _check_non_negative,
    _check_positive_int,
    _json_load,
    _json_save,
    _normalize_phase_vector,
)
from .types import CircuitData, FloatArray, GateParameters, Modes, ParameterValue

# A circuit's running state is Gaussian until it hits a gate that isn't --
# see `Circuit.run` and the non-Gaussian gates registered near the bottom
# of this module. There is deliberately no third, "back to Gaussian" case:
# once a computation is a `FockState`, it stays one (see `catsy.fock`).
CVState = GaussianState | FockState


class GateTransform(Protocol):
    """Callable contract for a state transformation.

    A single transform is expected to accept *either* representation and
    return the matching one (see the dispatching gates registered below,
    e.g. `catsy.optics.squeeze`) -- `Circuit` itself stays representation-
    agnostic and never inspects `state`'s type.
    """

    def __call__(
        self, state: CVState, modes: Modes, **kwargs: ParameterValue
    ) -> CVState: ...


@dataclass(frozen=True)
class Gate:
    """One fully bound transformation over named modes."""

    name: str
    transform: GateTransform
    modes: Modes
    kwargs: GateParameters

    def apply(self, state: Any | None) -> CVState:
        # `state` is None only when this is (or precedes) an InitialState
        # gate; Circuit.run() enforces that invariant before calling apply()
        # on any other gate, so the cast reflects an already-checked fact
        # rather than papering over an unchecked one.
        return self.transform(cast("CVState", state), self.modes, **self.kwargs)


class CircuitState(Protocol):
    """Minimal state interface required by :class:`Circuit`."""

    modes: Modes

    def reorder_modes(self, modes: Modes) -> Any: ...


@dataclass(frozen=True, eq=False, slots=True)
class Mode:
    """A named mode, optionally owned by the :class:`Circuit` that produced it.

    A ``Mode`` obtained from :meth:`Circuit.mode` is owned by that circuit,
    and can only be used to build gates on that same circuit: passing it to a
    different circuit, or mixing it into a gate on another circuit, is
    rejected before a :class:`Gate` is even constructed -- so a mode meant
    for one circuit can't accidentally end up wired into a different one.

    A ``Mode`` with ``owner=None`` is "free"/standalone: not tied to any
    circuit, e.g. for a one-off gate applied directly without building a
    :class:`Circuit` at all.

    Modes use object identity for equality: two `Mode("a")` instances are
    never equal to each other, owned or free, even with the same name --
    only a `Mode` obtained from `Circuit.mode(...)` is ever recognized as
    belonging to that circuit.
    """

    name: str
    owner: Circuit | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Mode name must be a non-empty string.")

    def __repr__(self) -> str:
        owned_by = f"owner={self.owner.name!r}" if self.owner is not None else "free"
        return f"Mode({self.name!r}, {owned_by})"


def _normalize_gate_kwargs(
    kwargs: dict[str, ParameterValue],
) -> dict[str, ParameterValue]:
    """Canonicalize any `alpha` in gate kwargs into real-valued `x`/`p`.

    `Gate.kwargs` is what `Circuit.to_dict()`/`save()` persists, and a raw
    complex amplitude isn't JSON serializable -- so this runs at gate
    *construction* time (in every place a `Gate` gets built from
    user-supplied kwargs), not just when the gate later runs. Only
    Displacer and InitialState(kind="coherent") ever pass `alpha`, and both
    transforms already accept `x`/`p` as an equivalent form (see `displace`
    and `initial_state` in `catsy.gaussian`), so this is a lossless,
    format-only substitution. A no-op when `alpha` isn't present.
    """
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
        # Re-derive `modes`/`_mode_registry` through `mode()` so a circuit
        # constructed with `Circuit(modes=(...))` (e.g. from `from_dict`)
        # gets the same owned-Mode handles and validation as one built up
        # via `.mode(...)`/`.add_mode(...)`.
        requested_modes, self.modes = self.modes, ()
        for mode_name in requested_modes:
            self.mode(mode_name)

    @property
    def gates(self) -> tuple[Gate, ...]:
        """The circuit's gates, in execution order. Read-only: use
        :meth:`add_gate` (or a fluent builder method) to append one."""
        return tuple(self._gates)

    @classmethod
    def register(cls, name: str, transform: GateTransform) -> None:
        """Register a gate name and its transformation for construction/loading."""
        if not name.strip():
            raise ValueError("Gate name must be a non-empty string.")
        _GATE_DESERIALIZERS[name] = transform

    def __getattr__(self, name: str) -> Any:
        """Expose registered gates as fluent circuit-building methods."""
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

        def apply(*modes: str | Mode, **kwargs: ParameterValue) -> Circuit:
            resolved_modes = tuple(self._resolve_mode(mode) for mode in modes)
            return self.add_gate(
                Gate(
                    name=gate_name,
                    transform=transform,
                    modes=resolved_modes,
                    kwargs=_normalize_gate_kwargs(kwargs),
                )
            )

        return apply

    def mode(self, mode_name: str) -> Mode:
        """Register a new mode on this circuit and return an owned handle for it.

        The returned :class:`Mode` is owned by this circuit. Fluent gate
        builder methods and :meth:`initial_state` accept either this handle
        or the bare mode-name string, but a `Mode` owned by a *different*
        circuit (or a free one, ``owner=None``) is rejected before a `Gate`
        is even constructed -- see :class:`Mode`.
        """
        if mode_name in self._mode_registry:
            raise ValueError(f"Mode '{mode_name}' is already registered in this circuit.")
        new_mode = Mode(name=mode_name, owner=self)
        self._mode_registry[mode_name] = new_mode
        self.modes = (*self.modes, mode_name)
        return new_mode

    def add_mode(self, mode_name: str) -> Circuit:
        """Fluent convenience wrapper around :meth:`mode` for chaining, e.g.
        ``Circuit().add_mode("a").add_mode("b")``. Discards the returned
        handle -- use :meth:`mode` directly when you need it."""
        self.mode(mode_name)
        return self

    def _resolve_mode(self, mode: str | Mode) -> str:
        """Validate one gate-target mode and return its plain name.

        A `Mode` must be owned by this circuit; a bare string must already
        be a registered mode name on this circuit. Either way, an
        unrecognized or wrongly-owned mode is rejected here, before a `Gate`
        is constructed -- not deferred to `run()`.
        """
        if isinstance(mode, Mode):
            if mode.owner is not self:
                owner_desc = (
                    "no circuit (a free/standalone mode)"
                    if mode.owner is None
                    else f"circuit {mode.owner.name!r}"
                )
                raise ValueError(
                    f"Mode {mode.name!r} belongs to {owner_desc}, not to this "
                    f"circuit ({self.name!r}). Use a Mode obtained from this "
                    "circuit's .mode(...)."
                )
            return mode.name
        if mode not in self._mode_registry:
            raise ValueError(
                f"Mode '{mode}' is not registered on this circuit. "
                f"Call circuit.mode({mode!r}) first."
            )
        return mode

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
        unknown_modes = [
            mode for mode in normalized_modes if mode not in self._mode_registry
        ]
        if unknown_modes:
            raise ValueError(
                f"{gate.name} targets mode(s) {unknown_modes!r} not registered on "
                f"this circuit ({self.name!r}); call circuit.mode(...) to register "
                "them first."
            )
        self._gates.append(gate)
        return self

    def initial_state(
        self, *modes: str | Mode, kind: str = "vacuum", **kwargs: ParameterValue
    ) -> Circuit:
        """Append the registered ``initial_state`` gate to the circuit."""
        try:
            transform = _GATE_DESERIALIZERS["InitialState"]
        except KeyError as exc:
            raise RuntimeError("No InitialState gate has been registered.") from exc
        resolved_modes = tuple(self._resolve_mode(mode) for mode in modes)
        return self.add_gate(
            Gate(
                name="InitialState",
                transform=transform,
                modes=resolved_modes,
                kwargs=_normalize_gate_kwargs({"kind": kind, **kwargs}),
            )
        )

    def run(self, initial_state: CVState | None = None) -> Any:
        """Run the ordered gate chain against an optional initial state."""
        if not self.modes:
            raise ValueError("Circuit has no registered modes.")

        current_state: Any = None
        if initial_state is not None:
            if set(initial_state.modes) != set(self.modes):
                raise ValueError("Initial state's modes don't match the circuit's modes.")
            current_state = initial_state.reorder_modes(self.modes)

        for idx, gate in enumerate(self._gates):
            # Mode registration/ownership is validated once, in add_gate();
            # every gate reaching this loop has already passed that check.
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

    def render_schematic(self, input_states: dict[str, str] | None = None) -> str:
        """Render the circuit as a deterministic plain-text schematic."""
        if not self._gates:
            return (
                f"┌─── {self.name} ───┐\n"
                "│   (Empty Circuit)   │\n"
                "└──────────────────────────┘"
            )

        ordered_modes = list(self.modes)
        position = {mode: i for i, mode in enumerate(ordered_modes)}
        input_states = input_states or {}

        lines = {
            mode: f"{input_states.get(mode, '|0>'):<9} ──[{mode}]──"
            for mode in ordered_modes
        }

        for gate in self._gates:
            label, block_width = _render_gate_label(gate)
            involved_modes = sorted(gate.modes, key=position.__getitem__)
            first_pos = position[involved_modes[0]]
            last_pos = position[involved_modes[-1]]

            for mode in ordered_modes:
                pos = position[mode]
                if mode in involved_modes:
                    if len(involved_modes) > 1:
                        if pos == first_pos:
                            connector = "┬"
                        elif pos == last_pos:
                            connector = "┴"
                        else:
                            connector = "┼"
                        cell = f"[{connector}{label.center(block_width - 2)}{connector}]"
                    else:
                        cell = f"[{label.center(block_width)}]"
                    lines[mode] += f"─{cell}─"
                elif len(involved_modes) > 1 and first_pos < pos < last_pos:
                    bridge = "│".center(block_width + 2, "─")
                    lines[mode] += f"─{bridge}─"
                else:
                    lines[mode] += "─" * (block_width + 4)

        title_banner = f"─┤ Schematic: {self.name} ├"
        out = [f"┌{title_banner:─<78}┐"]
        for mode in ordered_modes:
            out.append(f"│  {lines[mode]}── OUT  │")
        out.append("└" + "─" * 78 + "┘")
        return "\n".join(out)

    def draw(self, input_states: dict[str, str] | None = None) -> None:
        """Print the circuit schematic to stdout for interactive/notebook use."""
        print("\n" + self.render_schematic(input_states) + "\n")


# ---------------------------------------------------------------------------
# Universal gates: one dispatcher per physical operation, not per
# representation. Squeezing/rotation/displacement/beam splitters/loss are
# Gaussian (quadratic-generator) operations with an exact representation in
# either picture, so each dispatcher below just forwards to the matching
# GaussianState method (via the `catsy.gaussian` functions) or FockState
# method depending on what `state` currently is. This is what lets a
# `Circuit` mix these gates freely regardless of when/whether it has been
# promoted into Fock space (see the non-Gaussian gates further down).
# ---------------------------------------------------------------------------


def squeeze(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.squeeze(
            mode=modes[0],
            r=cast(float, kwargs["r"]),
            theta=cast(float, kwargs.get("theta", 0.0)),
        )
    return _gaussian_squeeze(state, modes, **kwargs)


def rotate(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.rotate(mode=modes[0], phi=cast(float, kwargs["phi"]))
    return _gaussian_rotate(state, modes, **kwargs)


def displace(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.displace(
            mode=modes[0],
            alpha=cast(complex, kwargs["alpha"]) if "alpha" in kwargs else None,
            x=cast(float, kwargs["x"]) if "x" in kwargs else None,
            p=cast(float, kwargs["p"]) if "p" in kwargs else None,
        )
    return _gaussian_displace(state, modes, **kwargs)


def beam_splitter(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.beam_splitter(
            mode_a=modes[0], mode_b=modes[1], eta=cast(float, kwargs["eta"])
        )
    return _gaussian_beam_splitter(state, modes, **kwargs)


def loss(state: CVState, modes: Modes, **kwargs: ParameterValue) -> CVState:
    if isinstance(state, FockState):
        return state.loss(mode=modes[0], eta=cast(float, kwargs["eta"]))
    return _gaussian_loss(state, modes, **kwargs)


def thermal_loss(
    state: CVState,
    modes: Modes,
    **kwargs: ParameterValue,
) -> CVState:
    if isinstance(state, FockState):
        return state.thermal_loss(
            mode=modes[0],
            eta=cast(float, kwargs["eta"]),
            nbar=cast(float, kwargs.get("nbar", 0.0)),
            ancilla_cutoff=(
                cast(int, kwargs["ancilla_cutoff"])
                if "ancilla_cutoff" in kwargs
                else None
            ),
        )

    return _gaussian_thermal_loss(state, modes, **kwargs)


# ---------------------------------------------------------------------------
# Non-Gaussian gates: Fock-only by physics (a ideal photon
# subtraction/addition is not a Gaussian channel), so unlike the dispatchers
# above these have no Gaussian branch. What they do have is automatic,
# one-way promotion: a `GaussianState` embeds *exactly* (up to Fock-space
# truncation) into a `FockState` via `to_fock`, so lifting into Fock space
# on first contact with one of these gates loses nothing and needs no
# permission from the caller -- it only needs to know the cutoff to embed
# into, via this gate's `N_cutoff` kwarg. There is deliberately no gate that
# goes the other way (see `catsy.fock`): once promoted, a circuit stays in
# Fock space for the rest of its run.
# ---------------------------------------------------------------------------


def _ensure_fock(state: CVState, kwargs: dict[str, ParameterValue]) -> FockState:
    if isinstance(state, FockState):
        return state
    if "N_cutoff" not in kwargs:
        raise ValueError(
            "This gate is non-Gaussian and the circuit hasn't been promoted "
            "into Fock space yet; pass N_cutoff=... to this gate call so it "
            "knows what cutoff to embed the current (still-Gaussian) state "
            "into."
        )
    return cast(GaussianState, state).to_fock(cast(int, kwargs["N_cutoff"]))


def photon_subtraction(
    state: CVState, modes: Modes, **kwargs: ParameterValue
) -> FockState:
    return _ensure_fock(state, kwargs).photon_subtraction(mode=modes[0])


def photon_addition(state: CVState, modes: Modes, **kwargs: ParameterValue) -> FockState:
    return _ensure_fock(state, kwargs).photon_addition(mode=modes[0])


def realistic_photon_subtraction(
    state: CVState, modes: Modes, **kwargs: ParameterValue
) -> FockState:
    return _ensure_fock(state, kwargs).realistic_photon_subtraction(
        mode=modes[0],
        tap_reflectivity=cast(float, kwargs.get("tap_reflectivity", 0.05)),
        detector_efficiency=cast(float, kwargs.get("detector_efficiency", 0.6)),
        ancilla_cutoff=cast(int, kwargs.get("ancilla_cutoff", 6)),
    )


def realistic_photon_addition(
    state: CVState, modes: Modes, **kwargs: ParameterValue
) -> FockState:
    return _ensure_fock(state, kwargs).realistic_photon_addition(
        mode=modes[0],
        coupling_strength=cast(float, kwargs.get("coupling_strength", 0.05)),
        detector_efficiency=cast(float, kwargs.get("detector_efficiency", 0.6)),
        ancilla_cutoff=cast(int, kwargs.get("ancilla_cutoff", 6)),
    )


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


for _name, _transform in (
    ("Squeezer", squeeze),
    ("Rotator", rotate),
    ("Displacer", displace),
    ("BeamSplitter", beam_splitter),
    ("Noise", loss),
    ("ThermalLoss", thermal_loss),
    ("InitialState", initial_state),
    ("PhotonSubtraction", photon_subtraction),
    ("PhotonAddition", photon_addition),
    ("RealisticPhotonSubtraction", realistic_photon_subtraction),
    ("RealisticPhotonAddition", realistic_photon_addition),
):
    Circuit.register(_name, _transform)


def _render_gate_label(gate: Gate) -> tuple[str, int]:
    """Pick the abbreviated gate type and parameter for one schematic block."""
    param_str = ""
    for key, symbol in (("eta", "η"), ("phi", "φ"), ("r", "r"), ("x", "x"), ("p", "p")):
        if key in gate.kwargs:
            value = gate.kwargs[key]
            param_str = (
                f" {symbol}={value:.2f}"
                if isinstance(value, float)
                else f" {symbol}={value}"
            )
            break
    type_name = _GATE_LABEL_ABBREVIATIONS.get(gate.name, gate.name[:5])
    label = f" {type_name}{param_str} "
    return label, max(len(label) + 2, 12)


# ---------------------------------------------------------------------------
# Physical-system simulations
# ---------------------------------------------------------------------------


class KerrCavity:
    """Driven, dissipative single-mode cavity with Kerr nonlinearity.

    Parameters
    ----------
    K:
        Kerr nonlinearity strength in ``K * a†² a²``.
    kappa:
        Cavity photon-loss rate.
    N_cutoff:
        Fock-space Hilbert-space dimension.
    """

    def __init__(self, K: float, kappa: float, N_cutoff: int):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        if not np.isfinite(K):
            raise ValueError(f"K must be finite, got {K!r}.")
        self.K = K
        self.kappa = kappa
        self.N_cutoff = N_cutoff

    def run(
        self,
        rho_init: qt.Qobj,
        tlist: FloatArray,
        amp: float,
        t0: float,
        sigma: float,
    ) -> list[qt.Qobj]:
        """Evolve ``rho_init`` under the driven Kerr-cavity master equation.

        ``amp``, ``t0`` and ``sigma`` define the Gaussian drive pulse.
        """
        tlist = np.asarray(tlist, dtype=float)
        if tlist.ndim != 1 or len(tlist) < 2:
            raise ValueError("tlist must be a 1D array with at least 2 time points.")
        if not np.all(np.isfinite(tlist)):
            raise ValueError("tlist must contain only finite values.")
        if not np.isfinite(amp):
            raise ValueError(f"amp must be finite, got {amp!r}.")
        if not np.isfinite(t0):
            raise ValueError(f"t0 must be finite, got {t0!r}.")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma must be > 0 and finite, got {sigma!r}.")

        a = qt.destroy(self.N_cutoff)
        H_kerr = self.K * a.dag() * a.dag() * a * a

        def pulse_shape(t: float, amp: float, t0: float, sigma: float) -> float:
            return float(amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2)))

        H_total = [H_kerr, [a + a.dag(), pulse_shape]]
        c_ops = [np.sqrt(self.kappa) * a] if self.kappa > 0 else []
        args = {"amp": amp, "t0": t0, "sigma": sigma}

        result = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args=args)
        states: list[qt.Qobj] = result.states
        return states


class ObservableScanData(TypedDict):
    theta: FloatArray
    n1: list[float]
    n2: list[float]
    parity1: list[float]


class MachZehnderInterferometer:
    """Two-mode Mach-Zehnder interferometer with a lossy phase-sensing arm.

    The interferometer owns its input state and physical parameters. A phase
    scan is performed with :meth:`scan`; plotting the resulting observables
    is a separate concern (see ``catsy.fock.mzi_visualization.plot_mzi_scan``,
    which takes this object directly) -- like every other physics object in
    this module (``Circuit``, ``Gate``, ``KerrCavity``), this class has no
    plotting methods and no matplotlib dependency of its own.

    Parameters
    ----------
    state:
        Input state of the single optical mode entering port 1. May be a ket
        or density matrix. Port 2 is initialized in vacuum.
    N_cutoff:
        Fock-space Hilbert-space dimension for each optical mode.
    kappa:
        Photon-loss rate in the lossy arm.
    loss_time:
        Fixed physical exposure time of the lossy arm. The loss is applied
        before the scanned phase, so its strength is independent of phase.

    Examples
    --------
    >>> mzi = MachZehnderInterferometer(rho, N_cutoff=20)
    >>> results = mzi.scan()
    """

    DEFAULT_NUM_PHASE_POINTS = 200

    def __init__(
        self,
        state: qt.Qobj,
        N_cutoff: int,
        kappa: float = 0.0,
        *,
        loss_time: float = 1.0,
    ):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        _check_non_negative(loss_time, "loss_time")

        if not isinstance(state, qt.Qobj):
            raise TypeError("state must be a QuTiP Qobj.")
        if not (state.isket or state.isoper):
            raise ValueError("state must be a ket or density matrix.")

        expected_shape = (N_cutoff, 1) if state.isket else (N_cutoff, N_cutoff)
        if state.shape != expected_shape:
            raise ValueError(
                f"state has shape {state.shape}, expected {expected_shape} "
                f"for N_cutoff={N_cutoff}."
            )

        self.state = state
        self.N_cutoff = N_cutoff
        self.kappa = kappa
        self.loss_time = loss_time
        self.results: ObservableScanData | None = None

    @classmethod
    def even_cat(
        cls,
        *,
        cutoff: int = 22,
        alpha: complex = 4.0 + 2j,
        kappa: float = 0.0,
        loss_time: float = 1.0,
    ) -> MachZehnderInterferometer:
        """Build an interferometer with an even cat state entering port 1."""
        return cls(
            make_even_cat(cutoff=cutoff, alpha=alpha),
            N_cutoff=cutoff,
            kappa=kappa,
            loss_time=loss_time,
        )

    def scan(self, theta_list: FloatArray | None = None) -> ObservableScanData:
        """Scan the phase of the lossy arm and return output observables.

        Parameters
        ----------
        theta_list:
            One-dimensional sequence of phase shifts. If omitted, 200
            uniformly spaced points between 0 and 2*pi are used.

        Returns
        -------
        ObservableScanData
            Phase values and the corresponding output-port photon numbers
            and parity. Also stored on ``self.results``.
        """
        if theta_list is None:
            theta_list = np.linspace(0.0, 2.0 * np.pi, self.DEFAULT_NUM_PHASE_POINTS)
        else:
            theta_list = np.asarray(theta_list, dtype=float)

        if theta_list.ndim != 1 or len(theta_list) < 1:
            raise ValueError("theta_list must be a non-empty 1D array.")
        if not np.all(np.isfinite(theta_list)):
            raise ValueError("theta_list must contain only finite values.")

        N = self.N_cutoff
        a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
        a2 = qt.tensor(qt.qeye(N), qt.destroy(N))

        n1_op = a1.dag() * a1
        n2_op = a2.dag() * a2
        parity1_op = (1j * np.pi * n1_op).expm()

        U_BS = ((1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())).expm()

        # Port 2 is initialized in vacuum.
        vacuum = qt.fock(N, 0)
        if self.state.isket:
            psi_in = qt.tensor(self.state, vacuum)
            psi_after_BS1 = U_BS * psi_in
        else:
            rho_in = qt.tensor(self.state, qt.ket2dm(vacuum))
            psi_after_BS1 = U_BS * rho_in * U_BS.dag()

        # Apply loss once, before the scanned phase.
        c_ops = (
            [np.sqrt(self.kappa) * a1]
            if self.kappa > 0.0 and self.loss_time > 0.0
            else []
        )
        if c_ops:
            loss_sim = qt.mesolve(
                0 * n1_op, psi_after_BS1, [0.0, self.loss_time], c_ops=c_ops
            )
            rho_after_loss = loss_sim.states[-1]
            if rho_after_loss.isket:
                rho_after_loss = qt.ket2dm(rho_after_loss)
        elif psi_after_BS1.isket:
            rho_after_loss = qt.ket2dm(psi_after_BS1)
        else:
            rho_after_loss = psi_after_BS1

        results: ObservableScanData = {
            "theta": theta_list,
            "n1": [],
            "n2": [],
            "parity1": [],
        }
        for theta in theta_list:
            U_phase = (1j * theta * n1_op).expm()
            rho_after_phase = U_phase * rho_after_loss * U_phase.dag()
            rho_out = U_BS * rho_after_phase * U_BS.dag()

            results["n1"].append(float(qt.expect(n1_op, rho_out)))
            results["n2"].append(float(qt.expect(n2_op, rho_out)))
            results["parity1"].append(float(qt.expect(parity1_op, rho_out).real))

        self.results = results
        return results

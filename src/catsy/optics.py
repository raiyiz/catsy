"""Reusable optical-bench layouts and QuTiP-based physical simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

from .core import _check_non_negative, _check_positive_int
from .gaussian import GaussianCircuit, GaussianState

# ---------------------------------------------------------------------------
# Component blueprint
# ---------------------------------------------------------------------------

# Structural contract for each component type.  Numerical/physical execution
# remains in ``GaussianOperations``; this table only defines what a layout
# component must look like before it can be registered.
_COMPONENT_SPECS = {
    "BeamSplitter": {"ports": 2, "kwargs": ("eta",)},
    "Loss": {"ports": 1, "kwargs": ("eta",)},
    "Squeezing": {"ports": 1, "kwargs": ("r", "theta")},
    "PhaseRotation": {"ports": 1, "kwargs": ("phi",)},
}


@dataclass
class OpticalComponent:
    """Blueprint for a component mapped to specific interface ports."""

    name: str  # e.g. "50:50 BS", "Fiber Loss", "Squeezer"
    op_type: str  # one of "BeamSplitter", "Loss", "Squeezing", "PhaseRotation"
    ports: tuple[str, ...]  # the ordered port/channel names it connects to
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the structural contract of a layout component.

        Component validation belongs here because an ``OpticalComponent`` is
        the reusable layout object.  The circuit compiler remains responsible
        for executing valid components.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OpticalComponent name must be a non-empty string.")

        if self.op_type not in _COMPONENT_SPECS:
            raise ValueError(
                f"Unknown optical component type {self.op_type!r}. "
                f"Known types: {sorted(_COMPONENT_SPECS)}."
            )

        self.ports = tuple(self.ports)
        self.kwargs = dict(self.kwargs)

        expected_ports = _COMPONENT_SPECS[self.op_type]["ports"]
        if len(self.ports) != expected_ports:
            raise ValueError(
                f"{self.op_type} requires exactly {expected_ports} port(s), "
                f"got {len(self.ports)}."
            )

        if len(set(self.ports)) != len(self.ports):
            raise ValueError(
                f"{self.op_type} cannot connect the same port more than once: "
                f"{self.ports!r}."
            )

        if any(not isinstance(port, str) or not port.strip() for port in self.ports):
            raise ValueError("All optical component ports must be non-empty strings.")

        expected_kwargs = _COMPONENT_SPECS[self.op_type]["kwargs"]
        if set(self.kwargs) != set(expected_kwargs):
            missing = sorted(set(expected_kwargs) - set(self.kwargs))
            extra = sorted(set(self.kwargs) - set(expected_kwargs))
            details = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"unexpected {extra}")
            raise ValueError(
                f"Invalid kwargs for {self.op_type}: " + ", ".join(details) + "."
            )

        for key in expected_kwargs:
            value = self.kwargs[key]
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(
                    f"{self.op_type} parameter {key!r} must be a finite scalar, "
                    f"got {value!r}."
                )

        if self.op_type in {"BeamSplitter", "Loss"}:
            eta = float(self.kwargs["eta"])
            if not 0.0 <= eta <= 1.0:
                raise ValueError(
                    f"{self.op_type} parameter 'eta' must be in [0, 1], got {eta}."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "ports": list(self.ports),
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpticalComponent:
        return cls(
            name=data["name"],
            op_type=data["op_type"],
            ports=tuple(data["ports"]),
            kwargs=data["kwargs"],
        )


# Maps an OpticalComponent.op_type to the GaussianCircuit builder method that
# implements it. Extend the component vocabulary by adding an entry here (and
# a matching `add_*`-style convenience method below) rather than touching
# `process_beam`.
_CIRCUIT_BUILDERS = {
    "BeamSplitter": lambda circuit, ports, kwargs: circuit.beam_splitter(
        ports[0], ports[1], **kwargs
    ),
    "Loss": lambda circuit, ports, kwargs: circuit.loss(ports[0], **kwargs),
    "Squeezing": lambda circuit, ports, kwargs: circuit.squeeze(ports[0], **kwargs),
    "PhaseRotation": lambda circuit, ports, kwargs: circuit.rotate(ports[0], **kwargs),
}

# Abbreviated labels and the kwarg used to render each component's parameter
# in `OpticalSetup.render_schematic`.
_TYPE_ABBREVIATIONS = {
    "BeamSplitter": "BS",
    "Loss": "LOSS",
    "Squeezing": "SQZ",
    "PhaseRotation": "PHASE",
}
_LABEL_PARAM_KEYS = ("eta", "phi", "r")


# ---------------------------------------------------------------------------
# Optical bench layout
# ---------------------------------------------------------------------------


class OpticalSetup:
    """A static, reusable layout of optical hardware components on a bench."""

    def __init__(self, name: str = "Custom Bench Layout"):
        self.name = name
        self.components: list[OpticalComponent] = []
        self.registered_ports: set[str] = set()

    def add_component(self, component: OpticalComponent) -> OpticalSetup:
        self.registered_ports.update(component.ports)
        self.components.append(component)
        return self

    # -- Syntactic sugar for quick assembly ---------------------------------

    def beam_splitter(
        self, name: str, port_a: str, port_b: str, eta: float = 0.5
    ) -> OpticalSetup:
        return self.add_component(
            OpticalComponent(name, "BeamSplitter", (port_a, port_b), {"eta": eta})
        )

    def fiber_loss(self, name: str, port: str, eta: float) -> OpticalSetup:
        return self.add_component(OpticalComponent(name, "Loss", (port,), {"eta": eta}))

    def inline_squeezer(
        self, name: str, port: str, r: float, theta: float = 0.0
    ) -> OpticalSetup:
        return self.add_component(
            OpticalComponent(name, "Squeezing", (port,), {"r": r, "theta": theta})
        )

    def phase_shifter(self, name: str, port: str, phi: float) -> OpticalSetup:
        return self.add_component(
            OpticalComponent(name, "PhaseRotation", (port,), {"phi": phi})
        )

    # -- Execution ------------------------------------------------------------

    def process_beam(self, input_state: GaussianState) -> GaussianState:
        """Runs a pre-built quantum state through this hardware layout."""
        if not self.components:
            raise ValueError(f"OpticalSetup '{self.name}' has no components to run.")

        circuit = GaussianCircuit()
        for mode in sorted(self.registered_ports):
            circuit.add_mode(mode)

        for comp in self.components:
            if comp.op_type not in _CIRCUIT_BUILDERS:
                raise KeyError(
                    f"Component '{comp.name}': unknown op_type '{comp.op_type}'. "
                    f"Known types: {sorted(_CIRCUIT_BUILDERS)}."
                )
            _CIRCUIT_BUILDERS[comp.op_type](circuit, comp.ports, comp.kwargs)

        return circuit.compile_and_run(initial_state=input_state)

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_name": self.name,
            "components": [c.to_dict() for c in self.components],
        }

    def save_layout(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_layout(cls, file_path: str | Path) -> OpticalSetup:
        data = json.loads(Path(file_path).read_text())
        setup = cls(name=data["layout_name"])
        for c in data["components"]:
            setup.add_component(OpticalComponent.from_dict(c))
        return setup

    # -- Visualization ----------------------------------------------------

    def render_schematic(self, input_states: dict[str, str] | None = None) -> str:
        """Renders the layout as a plain-text schematic, one line per port,
        left to right through the components in registration order.

        Pure and deterministic (no printing), so it's easy to assert on in
        tests; `draw` prints the result for interactive/notebook use.
        """
        if not self.components:
            return (
                f"┌─── {self.name} ───┐\n"
                f"│   (Empty Bench Layout)   │\n"
                f"└──────────────────────────┘"
            )

        ordered_ports = sorted(self.registered_ports)
        input_states = input_states or {}

        lines = {
            port: f"{input_states.get(port, '|0>'):<9} ──[{port}]──"
            for port in ordered_ports
        }

        for comp in self.components:
            label, block_width = self._render_label(comp)
            involved_ports = sorted(comp.ports)

            for port in ordered_ports:
                if port in involved_ports:
                    if len(involved_ports) > 1:
                        if port == involved_ports[0]:
                            connector = "┬"
                        elif port == involved_ports[-1]:
                            connector = "┴"
                        else:
                            connector = "┼"
                        cell = (
                            f"[{connector}{label.center(block_width - 2)}{connector}]"
                        )
                    else:
                        cell = f"[{label.center(block_width)}]"
                    lines[port] += f"─{cell}─"
                elif (
                    len(involved_ports) > 1
                    and involved_ports[0] < port < involved_ports[-1]
                ):
                    # A multi-port component spans over this port without
                    # touching it -- draw a vertical bridge, not a gap.
                    bridge = "│".center(block_width + 2, "─")
                    lines[port] += f"─{bridge}─"
                else:
                    lines[port] += "─" * (block_width + 4)

        title_banner = f"─┤ Schematic: {self.name} ├"
        out = [f"┌{title_banner:─<78}┐"]
        for port in ordered_ports:
            out.append(f"│  {lines[port]}── OUT  │")
        out.append("└" + "─" * 78 + "┘")
        return "\n".join(out)

    @staticmethod
    def _render_label(comp: OpticalComponent) -> tuple[str, int]:
        """Picks the abbreviated type name + parameter string used for one
        component's block in the schematic, and the block's display width."""
        param_str = ""
        for key, symbol in (("eta", "η"), ("phi", "φ"), ("r", "r")):
            if key in comp.kwargs:
                value = comp.kwargs[key]
                param_str = (
                    f" {symbol}={value:.2f}" if key == "phi" else f" {symbol}={value}"
                )
                break
        type_name = _TYPE_ABBREVIATIONS.get(comp.op_type, comp.op_type[:5])
        label = f" {type_name}{param_str} "
        return label, max(len(label) + 2, 12)

    def draw(self, input_states: dict[str, str] | None = None) -> None:
        """Prints the schematic to stdout for interactive/notebook use."""
        print("\n" + self.render_schematic(input_states) + "\n")


# ---------------------------------------------------------------------------
# Physical-system simulations
# ---------------------------------------------------------------------------
#
# These operate directly on QuTiP Fock-space states rather than on
# GaussianState/GaussianCircuit; they model specific pieces of optical
# hardware (a driven cavity, an interferometer) rather than generic
# phase-space transformations, which is why they live alongside
# OpticalSetup instead of in gaussian.py or fock.py.


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
        self.K = float(K)
        self.kappa = float(kappa)
        self.N_cutoff = N_cutoff

    def run(
        self,
        rho_init,
        tlist: np.ndarray,
        amp: float,
        t0: float,
        sigma: float,
    ) -> list:
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

        def pulse_shape(t, amp, t0, sigma):
            return amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

        H_total = [H_kerr, [a + a.dag(), pulse_shape]]
        c_ops = [np.sqrt(self.kappa) * a] if self.kappa > 0 else []
        args = {"amp": float(amp), "t0": float(t0), "sigma": float(sigma)}

        result = qt.mesolve(H_total, rho_init, tlist, c_ops=c_ops, args=args)
        return result.states


class MachZehnderInterferometer:
    """Two-mode Mach-Zehnder interferometer with a lossy phase-sensing arm.

    Parameters
    ----------
    kappa:
        Photon-loss rate in the lossy arm.
    N_cutoff:
        Fock-space Hilbert-space dimension for each optical mode.
    loss_time:
        Fixed physical exposure time of the lossy arm. The loss is applied
        before the scanned phase, so its strength is independent of phase.
    """

    def __init__(self, kappa: float, N_cutoff: int, *, loss_time: float = 1.0):
        _check_positive_int(N_cutoff, "N_cutoff")
        _check_non_negative(kappa, "kappa")
        _check_non_negative(loss_time, "loss_time")
        self.kappa = float(kappa)
        self.N_cutoff = N_cutoff
        self.loss_time = float(loss_time)

    def scan(self, psi_cat_single, theta_list: np.ndarray) -> dict:
        """Scan the phase of the lossy arm and return output observables.

        The model is input -> 50:50 beam splitter -> fixed-time amplitude
        damping on arm 1 -> phase shift on arm 1 -> second 50:50 beam splitter.
        The returned dictionary contains ``theta``, ``n1``, ``n2`` and
        ``parity1`` arrays.
        """
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

        psi_in = qt.tensor(psi_cat_single, qt.fock(N, 0))
        psi_after_BS1 = U_BS * psi_in

        c_ops = [np.sqrt(self.kappa) * a1] if self.kappa > 0 and self.loss_time > 0 else []
        if c_ops:
            loss_sim = qt.mesolve(
                0 * n1_op,
                psi_after_BS1,
                [0.0, self.loss_time],
                c_ops=c_ops,
            )
            rho_after_loss = loss_sim.states[-1]
            if rho_after_loss.isket:
                rho_after_loss = qt.ket2dm(rho_after_loss)
        elif psi_after_BS1.isket:
            rho_after_loss = qt.ket2dm(psi_after_BS1)
        else:
            rho_after_loss = psi_after_BS1

        results = {"theta": theta_list, "n1": [], "n2": [], "parity1": []}

        for theta in theta_list:
            U_phase = (1j * float(theta) * n1_op).expm()
            rho_after_phase = U_phase * rho_after_loss * U_phase.dag()
            rho_out = U_BS * rho_after_phase * U_BS.dag()

            results["n1"].append(qt.expect(n1_op, rho_out))
            results["n2"].append(qt.expect(n2_op, rho_out))
            results["parity1"].append(qt.expect(parity1_op, rho_out).real)

        return results

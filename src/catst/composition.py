"""Static optical-bench layouts.

`OpticalSetup` composes named hardware components (beam splitters, loss
channels, squeezers, phase shifters) into a reusable layout: build it once,
save it to JSON, reload it later, and run any `GaussianState` through it.

This is deliberately a different concern from `states.py`'s circuit
compiler: `OpticalSetup` is a *layout* -- named components wired to port
labels, independent of any particular input state -- while `GaussianCircuit`
is what actually executes a fixed sequence of gates over a fixed mode set.
`OpticalSetup.process_beam` bridges the two by compiling a throwaway
`GaussianCircuit` from the layout on each call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .states import GaussianCircuit, GaussianState

# ---------------------------------------------------------------------------
# Component blueprint
# ---------------------------------------------------------------------------


@dataclass
class OpticalComponent:
    """Blueprint for a component mapped to specific interface ports."""

    name: str  # e.g. "50:50 BS", "Fiber Loss", "Squeezer"
    op_type: str  # one of "BeamSplitter", "Loss", "Squeezing", "PhaseRotation"
    ports: tuple[str, ...]  # the ordered port/channel names it connects to
    kwargs: dict[str, Any] = field(default_factory=dict)

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
                        cell = f"[{connector}{label.center(block_width - 2)}{connector}]"
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

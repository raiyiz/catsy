import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class JournalMetadata:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Untitled Simulation"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SetupConfig:
    """Records the physical variants and setup configurations."""

    modes: list[str]
    initial_alphas: dict[str, list[float]]
    environment_noise: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunPayload:
    """Holds the raw resulting datasets, cast into serializable formats."""

    final_state_cv: dict[str, Any] | None = None
    time_series: dict[str, list[float]] = field(default_factory=dict)
    wigner_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class JournalEntry:
    metadata: JournalMetadata = field(default_factory=JournalMetadata)
    setup: SetupConfig = field(init=False)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    results: RunPayload = field(default_factory=RunPayload)

    def attach_circuit(self, circuit: Any) -> None:
        """Extracts definition parameters and sequence directly from a GaussianCircuit."""
        circ_dict = circuit.to_dict()
        self.setup = SetupConfig(
            modes=circ_dict["modes"], initial_alphas=circ_dict["initial_alphas"]
        )
        self.timeline = circ_dict["operations"]

    def log_result_array(self, key: str, array: np.ndarray) -> None:
        """Converts arbitrary numpy computation structures into JSON safe lists."""
        self.results.time_series[key] = array.tolist()

    def log_wigner(
        self, mode_name: str, W: np.ndarray, X: np.ndarray, P: np.ndarray
    ) -> None:
        """Stores calculated analytical phase-space slices."""
        self.results.wigner_data[mode_name] = {
            "W": W.tolist(),
            "X": X.tolist(),
            "P": P.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.__dict__,
            "setup": self.setup.__dict__,
            "timeline": self.timeline,
            "results": {
                "final_state_cv": self.results.final_state_cv,
                "time_series": self.results.time_series,
                "wigner_data": self.results.wigner_data,
            },
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / f"entry_{self.metadata.entry_id}.json"
        file_path.write_text(json.dumps(self.to_dict(), indent=2))
        return file_path


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImprovedJournalEntry:
    # 1. Globale Metadaten des Experiments
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Quantum Simulation Log"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # Erlaubt es, dasselbe Setup mit unterschiedlichen Sweeps im selben Journal zu speichern
    runs: list[dict[str, Any]] = field(default_factory=list)

    def log_simulation_run(
        self,
        run_name: str,
        setup_layout_file: str | Path,
        metrics: dict[str, Any],
        arrays: dict[str, dict[str, Any]],
    ) -> None:
        """
        Logs a tightly bound dataset representing one execution of a setup.

        Args:
            run_name: Name of this sweep/run (e.g., "Squeezing Sweep r=0.5")
            setup_layout_file: Path to the JSON layout file used for this run
            metrics: Single-value results (e.g., {"purity": 0.98, "duan_score": 1.45})
            arrays: Dict of dicts containing the heavy data + metadata, e.g.:
                    {"wigner": {"data": [...], "unit": "Probability Density", "axes": ["x", "p"]}}
        """
        run_entry = {
            "run_id": str(uuid.uuid4()),
            "run_name": run_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "hardware_layout_reference": str(setup_layout_file),
            "scalar_results": metrics,
            "data_payloads": {},
        }

        # Jedes Array wird explizit mit seinen Metadaten (Einheiten, Dimensionen) verpackt
        for key, payload in arrays.items():
            run_entry["data_payloads"][key] = {
                "values": payload["data"],
                "unit": payload.get("unit", "arbitrary_units"),
                "dimensions": payload.get("dimensions", []),
            }

        self.runs.append(run_entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0.0",
            "metadata": {
                "entry_id": self.entry_id,
                "title": self.title,
                "timestamp": self.timestamp,
                "tags": self.tags,
                "notes": self.notes,
            },
            "simulations": self.runs,
        }

    def save(self, directory: Path) -> Path:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"journal_{self.entry_id}.json"
        file_path.write_text(json.dumps(self.to_dict(), indent=2))
        return file_path


class SimulationJournal:
    """Manages index directories containing distinct serialized simulation JSON files."""

    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def new_entry(
        self, title: str, tags: list[str] = None, notes: str = ""
    ) -> JournalEntry:
        entry = JournalEntry()
        entry.metadata.title = title
        entry.metadata.tags = tags or []
        entry.metadata.notes = notes
        return entry

    def fetch_history_summary(self) -> list[dict[str, Any]]:
        """Scans folder index metrics without hydrating large data arrays into memory."""
        summaries = []
        for file in self.storage_path.glob("*.json"):
            with open(file, "r") as f:
                raw = json.load(f)
                meta = raw["metadata"]
                summaries.append(
                    {
                        "entry_id": meta["entry_id"],
                        "title": meta["title"],
                        "timestamp": meta["timestamp"],
                        "tags": meta["tags"],
                        "file_path": str(file),
                    }
                )
        return sorted(summaries, key=lambda x: x["timestamp"], reverse=True)


# Create a new module: catst/composition.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

from catst.states import GaussianCircuit, GaussianState


@dataclass
class OpticalComponent:
    """Blueprint for a component mapped to specific interface lines."""

    name: str  # e.g., "50:50 BS", "Fiber Loss", "Squeezer"
    op_type: str  # Maps to "BeamSplitter", "Loss", "Squeezing", etc.
    ports: tuple[str, ...]  # The ordered mode/channel names it connects to
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "ports": list(self.ports),
            "kwargs": self.kwargs,
        }


class OpticalSetup:
    """Represents a static layout of optical hardware components on a bench."""

    def __init__(self, name: str = "Custom Bench Layout"):
        self.name = name
        self.components: list[OpticalComponent] = []
        self.registered_ports: set[str] = set()

    def add_component(self, component: OpticalComponent) -> "OpticalSetup":
        for port in component.ports:
            self.registered_ports.add(port)
        self.components.append(component)
        return self

    # --- Syntactic Sugar for Quick Assembly ---
    def beam_splitter(
        self, name: str, port_a: str, port_b: str, eta: float = 0.5
    ) -> "OpticalSetup":
        return self.add_component(
            OpticalComponent(name, "BeamSplitter", (port_a, port_b), {"eta": eta})
        )

    def fiber_loss(self, name: str, port: str, eta: float) -> "OpticalSetup":
        return self.add_component(OpticalComponent(name, "Loss", (port,), {"eta": eta}))

    def inline_squeezer(
        self, name: str, port: str, r: float, theta: float = 0.0
    ) -> "OpticalSetup":
        return self.add_component(
            OpticalComponent(name, "Squeezing", (port,), {"r": r, "theta": theta})
        )

    def phase_shifter(self, name: str, port: str, phi: float) -> "OpticalSetup":
        return self.add_component(
            OpticalComponent(name, "PhaseRotation", (port,), {"phi": phi})
        )

    # --- Execution Engine ---
    def process_beam(self, input_state: GaussianState) -> GaussianState:
        """Injects a pre-built quantum state directly into this hardware layout."""
        circuit = GaussianCircuit()
        # Ensure all physical line modes exist in the target circuit
        for mode in sorted(self.registered_ports):
            circuit.add_mode(mode)

        # Dynamically compile the static layout steps
        for comp in self.components:
            if comp.op_type == "BeamSplitter":
                circuit.beam_splitter(comp.ports[0], comp.ports[1], **comp.kwargs)
            elif comp.op_type == "Loss":
                circuit.loss(comp.ports[0], **comp.kwargs)
            elif comp.op_type == "Squeezing":
                circuit.squeeze(comp.ports[0], **comp.kwargs)
            elif comp.op_type == "PhaseRotation":
                circuit.rotate(comp.ports[0], **comp.kwargs)

        return circuit.compile_and_run(initial_state=input_state)

    # --- Serialization ---
    def save_layout(self, file_path: str | Path) -> None:
        layout_data = {
            "layout_name": self.name,
            "components": [c.to_dict() for c in self.components],
        }
        Path(file_path).write_text(json.dumps(layout_data, indent=2))

    @classmethod
    def load_layout(cls, file_path: str | Path) -> "OpticalSetup":
        data = json.loads(Path(file_path).read_text())
        setup = cls(name=data["layout_name"])
        for c in data["components"]:
            setup.add_component(
                OpticalComponent(
                    c["name"], c["op_type"], tuple(c["ports"]), c["kwargs"]
                )
            )
        return setup

    def draw(self, input_states: dict[str, str] = None) -> None:
        """
        Prints an advanced structural schematic with full ANSI terminal color mapping.
        """
        # ANSI Escape Codes for crisp terminal colors
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"
        C_GRAY = "\033[90m"
        C_RED = "\033[91m"
        C_GREEN = "\033[92m"
        C_YELLOW = "\033[93m"
        C_BLUE = "\033[94m"
        C_MAG = "\033[95m"
        C_CYAN = "\033[96m"

        if not self.components:
            print(
                f"┌─── {self.name} ───┐\n│   (Empty Bench Layout)   │\n└──────────────────────────┘"
            )
            return

        ordered_ports = sorted(list(self.registered_ports))
        input_states = input_states or {}

        # 1. Initialize ports with colored state vectors and bracketed names
        diagram_lines = {}
        for port in ordered_ports:
            state_label = input_states.get(port, "|0>")
            # Highlight state vectors in Cyan, and ports in bold Magenta
            diagram_lines[port] = (
                f"{C_CYAN}{state_label:<9}{C_RESET} {C_GRAY}──{C_RESET}[{C_MAG}{C_BOLD}{port}{C_RESET}]"
            )

        # 2. Process operations and build structural blocks
        for comp in self.components:
            # Dynamic parameter mapping and color selection based on hardware type
            param_str = ""
            if "eta" in comp.kwargs:
                param_str = f" η={comp.kwargs['eta']}"
                comp_color = C_BLUE
                type_name = "BS"
            elif "phi" in comp.kwargs:
                param_str = f" φ={comp.kwargs['phi']:.2f}"
                comp_color = C_YELLOW
                type_name = "PHASE"
            elif "r" in comp.kwargs:
                param_str = f" r={comp.kwargs['r']}"
                comp_color = C_GREEN
                type_name = "SQZ"
            elif "eta" in comp.kwargs or comp.op_type == "Loss":
                param_str = f" η={comp.kwargs.get('eta', '')}"
                comp_color = C_RED
                type_name = "LOSS"
            else:
                comp_color = C_RESET
                type_name = comp.op_type[:5]

            label = f" {type_name}{param_str} "
            block_width = max(len(label) + 2, 12)
            involved_ports = sorted(list(comp.ports))

            for port in ordered_ports:
                if port in involved_ports:
                    if len(involved_ports) > 1:
                        # Multi-port rails (e.g. BeamSplitters)
                        if port == involved_ports:
                            connector = "┬"
                        elif port == involved_ports[-1]:
                            connector = "┴"
                        else:
                            connector = "┼"

                        # Apply color exclusively to the block boundary and text content
                        padding = block_width - 2
                        centered_label = label.center(padding)
                        cell = f"{C_GRAY}─{C_RESET}{comp_color}[{connector}{C_BOLD}{centered_label}{C_RESET}{comp_color}]{C_RESET}"
                        diagram_lines[port] += cell
                    else:
                        # Single-port rail
                        centered_label = label.center(block_width)
                        cell = f"{C_GRAY}─{C_RESET}{comp_color}[{C_BOLD}{centered_label}{C_RESET}{comp_color}]{C_RESET}"
                        diagram_lines[port] += cell
                else:
                    # Logic for drawing vertical cross-over bridges when a wire is bypassed
                    if (
                        len(involved_ports) > 1
                        and involved_ports < port < involved_ports[-1]
                    ):
                        bridge = f"{comp_color}│{C_RESET}".center(
                            block_width + len(comp_color) + len(C_RESET), "─"
                        )
                        diagram_lines[port] += f"{C_GRAY}─{bridge}─{C_RESET}"
                    else:
                        diagram_lines[port] += (
                            f"{C_GRAY}" + "─" * (block_width + 2) + f"{C_RESET}"
                        )

        # 3. Print out execution frame canvas
        title_banner = f" ┤ Schematic: {self.name} ├ "
        print(f"\n{C_GRAY}┌{title_banner:─<88}┐{C_RESET}")
        for port in ordered_ports:
            print(
                f"{C_GRAY}│{C_RESET}  {diagram_lines[port]}{C_GRAY}───{C_RESET} {C_MAG}{C_BOLD}OUT{C_RESET}  {C_GRAY}│{C_RESET}"
            )
        print(f"{C_GRAY}└" + "─" * 88 + f"┘{C_RESET}\n")

    def old_draw(self, input_states: dict[str, str] = None) -> None:
        """
        Prints an advanced structural schematic of the optical circuit layout.
        Args:
            input_states: Optional dict mapping port_name -> state_label (e.g. {"line_1": "|α=2.0>"})
        """
        if not self.components:
            print(
                f"┌─── {self.name} ───┐\n│   (Empty Bench Layout)   │\n└──────────────────────────┘"
            )
            return

        ordered_ports = sorted(list(self.registered_ports))
        input_states = input_states or {}

        # 1. Initialize ports with their input state representations
        diagram_lines = {}
        for port in ordered_ports:
            state_label = input_states.get(port, "|0>")  # Default to vacuum state
            diagram_lines[port] = f"{state_label:<9} ──[{port}]──"

        # 2. Process operations and build structural blocks
        for comp in self.components:
            # Extract key parameter values for compact technical rendering
            param_str = ""
            if "eta" in comp.kwargs:
                param_str = f" η={comp.kwargs['eta']}"
            elif "phi" in comp.kwargs:
                param_str = f" φ={comp.kwargs['phi']:.2f}"
            elif "r" in comp.kwargs:
                param_str = f" r={comp.kwargs['r']}"

            # Formatting abbreviations
            type_abbr = {
                "BeamSplitter": "BS",
                "Loss": "LOSS",
                "Squeezing": "SQZ",
                "PhaseRotation": "PHASE",
            }
            label = f" {type_abbr.get(comp.op_type, comp.op_type)}{param_str} "
            block_width = max(len(label) + 2, 12)

            # Determine vertical spans for multi-port connections (like BeamSplitters)
            involved_ports = sorted(list(comp.ports))

            for port in ordered_ports:
                if port in involved_ports:
                    if len(involved_ports) > 1:
                        # Multi-port logic using elegant vertical bus couplings
                        if port == involved_ports[0]:
                            connector = "┬"
                        elif port == involved_ports[-1]:
                            connector = "┴"
                        else:
                            connector = "┼"

                        # Pad the block to keep grid alignments locked
                        cell = f"[{connector}{label.center(block_width-2)}{connector}]"
                        diagram_lines[port] += f"─{cell}─"
                    else:
                        # Single port components
                        cell = f"[{label.center(block_width)}]"
                        diagram_lines[port] += f"─{cell}─"
                else:
                    # Pass-through line (empty space bridge)
                    # If a multi-port component passes *over* this row, draw a vertical bridge
                    if (
                        len(involved_ports) > 1
                        and involved_ports[0] < port < involved_ports[-1]
                    ):
                        bridge = "│".center(block_width + 2, "─")
                        diagram_lines[port] += f"─{bridge}─"
                    else:
                        diagram_lines[port] += "─" * (block_width + 4)

        # 3. Print the finalized high-fidelity canvas
        title_banner = f"─┤ Schematic: {self.name} ├"
        print(f"\n┌{title_banner:─<78}┐")
        for port in ordered_ports:
            print(f"│  {diagram_lines[port]}── OUT  │")
        print("└" + "─" * 78 + "┘\n")

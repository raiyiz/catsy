#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 8
// ==========================================
= Chapter 8: Optical Bench Layouts (`OpticalSetup`)

`GaussianCircuit` (Chapter 3) describes a *gate sequence* — abstract and independent of a particular laboratory layout. `optics.py` adds a layer on top that models a reusable, named *hardware bench layout*: components with fixed ports that can be run repeatedly against different input states, saved/loaded, and visualized as a text schematic. The physical component vocabulary is consistent with standard linear/Gaussian optical processing; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)].

== The component blueprint (`OpticalComponent`)

Every component is a structurally validated tuple of name, type, port list, and parameters. The allowed vocabulary and signature catalog is declared centrally in `_COMPONENT_SPECS`:

```python
_COMPONENT_SPECS = {
    "BeamSplitter": {"ports": 2, "kwargs": ("eta",)},
    "Loss": {"ports": 1, "kwargs": ("eta",)},
    "Squeezing": {"ports": 1, "kwargs": ("r", "theta")},
    "PhaseRotation": {"ports": 1, "kwargs": ("phi",)},
}
```

`OpticalComponent.__post_init__` enforces against this table, among other things: a valid, known `op_type`; exactly the expected number of ports with no duplicates; exactly the expected parameter set (no missing, no extra `kwargs`); finite scalar parameter values; and $eta in [0,1]$ for beam splitters and loss elements. The separation of responsibilities is important here: `OpticalComponent` validates *only* the structural correctness of the component as a blueprint — the actual numerical execution remains entirely the responsibility of `GaussianOperations` (Chapter 2).

== Declaratively assembling a bench (`OpticalSetup`)

`OpticalSetup` collects components in registration order and offers a fluent, chainable builder API for doing so:

```python
def beam_splitter(
    self, name: str, port_a: str, port_b: str, eta: float = 0.5
) -> OpticalSetup:
    return self.add_component(
        OpticalComponent(name, "BeamSplitter", (port_a, port_b), {"eta": eta})
    )

def fiber_loss(self, name: str, port: str, eta: float) -> OpticalSetup:
    return self.add_component(OpticalComponent(name, "Loss", (port,), {"eta": eta}))
```

Every call returns `self`, so a setup can be expressed as a readable chain, e.g. `OpticalSetup("Bench A").inline_squeezer("SQZ1", "a", r=0.6).beam_splitter("BS1", "a", "b")`.

== Execution: from blueprint to `GaussianCircuit`

`process_beam` translates the static layout into a concrete `GaussianCircuit` (Chapter 3) and runs it against a supplied input state. The translation happens via a second dispatch table, `_CIRCUIT_BUILDERS`, which maps every `op_type` to the matching `GaussianCircuit` builder method:

```python
_CIRCUIT_BUILDERS = {
    "BeamSplitter": lambda circuit, ports, kwargs: circuit.beam_splitter(
        ports[0], ports[1], **kwargs
    ),
    "Loss": lambda circuit, ports, kwargs: circuit.loss(ports[0], **kwargs),
    "Squeezing": lambda circuit, ports, kwargs: circuit.squeeze(ports[0], **kwargs),
    "PhaseRotation": lambda circuit, ports, kwargs: circuit.rotate(ports[0], **kwargs),
}

def process_beam(self, input_state: GaussianState) -> GaussianState:
    if not self.components:
        raise ValueError(f"OpticalSetup '{self.name}' has no components to run.")

    circuit = GaussianCircuit()
    for mode in sorted(self.registered_ports):
        circuit.add_mode(mode)

    for comp in self.components:
        if comp.op_type not in _CIRCUIT_BUILDERS:
            raise KeyError(...)
        _CIRCUIT_BUILDERS[comp.op_type](circuit, comp.ports, comp.kwargs)

    return circuit.compile_and_run(initial_state=input_state)
```

This pattern deliberately mirrors the `OPERATION_REGISTRY` from Chapter 3: extending the component vocabulary (a new `op_type` in `_COMPONENT_SPECS` plus a matching entry in `_CIRCUIT_BUILDERS`) does not touch `process_beam` itself. Since `process_beam` builds a fresh `GaussianCircuit` on every call, the same `OpticalSetup` instance can be run any number of times — even concurrently — against different input states without calls interfering with each other.

== Serialization and text schematic

Like `GaussianCircuit`, `OpticalSetup` is JSON-persistent via `to_dict`/`save_layout`/`load_layout`. In addition, `render_schematic` provides a pure, deterministic text visualization of the setup (one port per line, components left to right in registration order), which `draw` simply passes to `print`. The purity of `render_schematic` — no side effects, just a returned string — makes the schematic directly assertable in tests, while `draw` serves the interactive notebook use case.

---


== Scientific literature
The component-level interpretation of beam splitters, phase rotations, squeezing, and loss is standard Gaussian/linear quantum optics. Useful references are:

- #link("https://doi.org/10.1103/RevModPhys.84.621")[C. Weedbrook et al., “Gaussian quantum information,” *Reviews of Modern Physics* 84, 621–669 (2012).]
- #link("https://doi.org/10.1103/RevModPhys.77.513")[S. L. Braunstein and P. van Loock, “Quantum information with continuous variables,” *Reviews of Modern Physics* 77, 513–577 (2005).]
- #link("https://doi.org/10.1002/3527602976.ch13")[W. P. Schleich, *Quantum Optics in Phase Space*, chapter on optical interferometry (Wiley-VCH, 2001).]

The `OpticalSetup` and serialization abstractions themselves are software-design decisions of `catsy`; the references above support the physical meaning of the modeled optical components.

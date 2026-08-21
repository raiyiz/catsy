#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 8
// ==========================================
= Chapter 8: Optical Bench Layouts (`OpticalSetup`)

`Circuit` (Chapter 3) describes a *gate sequence* — abstract and independent of a particular laboratory layout. `optics.py` adds a layer on top that models a reusable, named *hardware bench layout*: components with fixed ports that can be run repeatedly against different input states, saved/loaded, and visualized as a text schematic. The physical component vocabulary is consistent with standard linear/Gaussian optical processing; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)].

== The component blueprint (`OpticalComponent`)

Every component is a named physical wrapper around one executable Gaussian callable. The component stores the callable itself as `operation`, its target ports, and its parameters; the optical layer adds the physical component name and layout semantics. This avoids a separate `CircuitOperation` representation and means the component can attach its operation directly to the setup's circuit. The callable is the executable contract and carries an explicit `name` attribute; serialization stores only that name.

== Declaratively assembling a bench (`OpticalSetup`)

`OpticalSetup` collects components in registration order and offers a fluent, chainable builder API for doing so:

```python
def beam_splitter(
    self, name: str, port_a: str, port_b: str, eta: float = 0.5
) -> OpticalSetup:
    return self.add_component(
        OpticalComponent(
            name, beam_splitter, (port_a, port_b), {"eta": eta}
        )
    )

def fiber_loss(self, name: str, port: str, eta: float) -> OpticalSetup:
    return self.add_component(
        OpticalComponent(name, loss, (port,), {"eta": eta})
    )
```

Every call returns `self`, so a setup can be expressed as a readable chain, e.g. `OpticalSetup("Bench A").inline_squeezer("SQZ1", "a", r=0.6).beam_splitter("BS1", "a", "b")`.

== Execution: `OpticalSetup` owns a `Circuit`

`OpticalSetup` composes with `Circuit` rather than translating a stored layout into a new circuit every time `process_beam` is called. The circuit is injected optionally and is created by default, following the same composition pattern as `Vehicle(circuit=Circuit())`:

```python
setup = OpticalSetup("Bench A", circuit=Circuit())
```

Each `OpticalComponent` contains its executable callable, and `add_component` attaches that same callable directly to the setup's circuit. There is no second operation representation and no component-to-operation conversion step:

```python
def add_component(self, component: OpticalComponent) -> OpticalSetup:
    self.registered_ports.update(component.ports)
    for port in component.ports:
        if port not in self.circuit.modes:
            self.circuit.add_mode(port)
    self.components.append(component)
    component.apply_to(self.circuit)
    return self
```

`process_beam` is consequently just the execution boundary:

```python
def process_beam(self, input_state: GaussianState) -> GaussianState:
    if not self.components:
        raise ValueError(f"OpticalSetup '{self.name}' has no components to run.")
    return self.circuit.run(input_state)
```

This removes the second `_CIRCUIT_BUILDERS` dispatch table and, more importantly, removes the duplicate executable representation. `Circuit` remains the single owner of Gaussian execution, while `OpticalSetup` retains the physical component metadata used for layout, serialization, and schematic rendering. The injected circuit is not serialized as part of the layout; loading a layout constructs its normal default circuit and repopulates it through `add_component`.

== Serialization and text schematic

Like `Circuit`, `OpticalSetup` is JSON-persistent via `to_dict`/`save_layout`/`load_layout`. In addition, `render_schematic` provides a pure, deterministic text visualization of the setup (one port per line, components left to right in registration order), which `draw` simply passes to `print`. The purity of `render_schematic` — no side effects, just a returned string — makes the schematic directly assertable in tests, while `draw` serves the interactive notebook use case.

---


=== Literature
The component-level interpretation of beam splitters, phase rotations, squeezing, and loss is standard Gaussian/linear quantum optics, covered by the same background literature as Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005), plus specifically:

- #link("https://doi.org/10.1002/3527602976.ch13")[W. P. Schleich, *Quantum Optics in Phase Space*, chapter on optical interferometry (Wiley-VCH, 2001).]

The `OpticalSetup` and serialization abstractions themselves are software-design decisions of `catsy`; the references above support the physical meaning of the modeled optical components.

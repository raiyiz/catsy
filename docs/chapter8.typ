#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 8
// ==========================================
= Chapter 8: Optical Bench Layouts (`OpticalSetup`)

`Circuit` (Chapter 3) describes an *ordered gate sequence* — abstract and independent of a particular laboratory layout. `optics.py` adds `OpticalSetup`, a reusable named hardware-bench layout whose gates have fixed modes and can be run repeatedly against different input states, saved/loaded, and visualized as a text schematic. The physical gate vocabulary is consistent with standard linear/Gaussian optical processing; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)].

== Gates in an optical setup

There is no separate component abstraction. A `Gate` already contains everything needed to describe one concrete optical transformation:

```python
Gate(
    name="BeamSplitter",
    transform=beam_splitter,
    modes=("a", "b"),
    kwargs={"eta": 0.5},
)
```

The `name` is the noun used as the stable gate identity and serialization key. The `transform` is the executable mathematical function; `modes` identify where it acts; and `kwargs` contain the parameters bound to this particular gate instance.

== Declaratively assembling a bench (`OpticalSetup`)

`OpticalSetup` stores Gates in registration order and offers a fluent, chainable builder API:

```python
setup = (
    OpticalSetup("Bench A")
    .inline_squeezer("a", r=0.6)
    .beam_splitter("a", "b", eta=0.5)
    .fiber_loss("b", eta=0.9)
)
```

Each convenience method constructs one fully bound `Gate` and attaches it to the setup. The same Gate can also be constructed explicitly and passed to `add_gate`:

```python
gate = Gate(
    name="Squeezer",
    transform=squeeze,
    modes=("a",),
    kwargs={"r": 0.6, "theta": 0.0},
)
setup.add_gate(gate)
```

No separate component object is required, and the setup does not maintain a second representation of the transformation.

== Execution: `OpticalSetup` owns a `Circuit`

`OpticalSetup` composes with `Circuit` rather than translating a stored layout into a new circuit every time `process_beam` is called. The circuit is injected optionally and is created by default:

```python
setup = OpticalSetup("Bench A", circuit=Circuit())
```

`add_gate` registers the Gate's modes with the setup and attaches the exact same Gate instance to the owned circuit:

```python
def add_gate(self, gate: Gate) -> OpticalSetup:
    self.registered_ports.update(gate.modes)
    for mode in gate.modes:
        if mode not in self.circuit.modes:
            self.circuit.add_mode(mode)
    self.gates.append(gate)
    self.circuit.add_gate(gate)
    return self
```

`process_beam` is the execution boundary:

```python
def process_beam(self, input_state: GaussianState) -> GaussianState:
    if not self.gates:
        raise ValueError(f"OpticalSetup '{self.name}' has no gates to run.")
    return self.circuit.run(input_state)
```

`Circuit` remains the single owner of sequential Gaussian execution. `OpticalSetup` adds only the physical layout concerns: registered modes, gate ordering, persistence, and schematic rendering.

== Serialization and text schematic

Like `Circuit`, `OpticalSetup` is JSON-persistent via `to_dict`/`save_layout`/`load_layout`. The serialized layout stores its ordered Gates using the same `gate`/`modes`/`kwargs` representation as a Circuit. `render_schematic` provides a pure, deterministic text visualization of the setup (one mode per line, gates left to right in registration order), which `draw` simply passes to `print`.

---


=== Literature
The component-level interpretation of beam splitters, phase rotations, squeezing, and loss is standard Gaussian/linear quantum optics, covered by the same background literature as Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005), plus specifically:

- #link("https://doi.org/10.1002/3527602976.ch13")[W. P. Schleich, *Quantum Optics in Phase Space*, chapter on optical interferometry (Wiley-VCH, 2001).]

The `OpticalSetup` and serialization abstractions themselves are software-design decisions of `catsy`; the references above support the physical meaning of the modeled optical components.

#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 8
// ==========================================
= Chapter 8: Reusable Layouts and Circuit Schematics

Chapter 3 introduced `Circuit` as an *ordered gate sequence*. This chapter covers the parts of `Circuit` aimed specifically at building and inspecting reusable hardware-style layouts: the fluent builder API, repeated execution against different input states, save/load, and the plain-text schematic renderer. None of this lives in a separate class -- there is no optical-bench abstraction distinct from `Circuit` itself, so a circuit built for an abstract Gaussian-gate sequence and one meant to model a physical bench are the same object and share the same API. The physical gate vocabulary is consistent with standard linear/Gaussian optical processing; see #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)].

== Gates in a circuit

There is no separate component abstraction. A `Gate` already contains everything needed to describe one concrete optical transformation:

```python
Gate(
    name="BeamSplitter",
    transform=beam_splitter,
    modes=("a", "b"),
    kwargs={"eta": 0.5},
)
```

The `name` is the noun used as the stable gate identity and serialization key. The `transform` is the executable mathematical function; `modes` identify where it acts; and `kwargs` contain the parameters bound to this particular gate instance. `Circuit` places no restriction on which registered gate names appear together in one circuit -- a circuit is free to mix `Displacer`, `ThermalLoss`, or an explicit `InitialState` gate alongside the passive/lossy gates below; nothing distinguishes a "physically buildable bench" from any other gate sequence.

== Declaratively assembling a layout

Every registered gate is available as a fluent, chainable builder method directly on `Circuit`:

```python
circuit = (
    Circuit(name="Bench A")
    .add_mode("a").add_mode("b")
    .squeeze("a", r=0.6)
    .beam_splitter("a", "b", eta=0.5)
    .loss("b", eta=0.9)
)
```

Each convenience method constructs one fully bound `Gate` and appends it to the circuit -- see Chapter 3 for how `__getattr__` resolves `circuit.squeeze(...)` to the registered `Squeezer` transform. The same Gate can also be constructed explicitly and passed to `add_gate`:

```python
gate = Gate(
    name="Squeezer",
    transform=squeeze,
    modes=("a",),
    kwargs={"r": 0.6, "theta": 0.0},
)
circuit.add_gate(gate)
```

No separate component object is required, and the circuit does not maintain a second representation of the transformation. Note that `add_gate` does not automatically register a gate's modes with the circuit -- call `add_mode` first, as above; an unregistered mode is only caught once the circuit runs (Chapter 3).

== Execution

A `Circuit` can be run repeatedly against different input states without rebuilding its gate sequence -- `run` never mutates `self._gates`:

```python
first = circuit.run(input_state_a)
second = circuit.run(input_state_b)
```

An empty circuit (no gates) simply returns whatever state it's given unchanged, rather than raising -- `run`'s only hard requirements are that the circuit has at least one registered mode, and that it ends up with a state from somewhere (either a passed-in `initial_state`, or its own `InitialState` gate).

== Serialization and text schematic

`Circuit` is JSON-persistent via `to_dict`/`from_dict`/`save`/`load` (Chapter 3). `render_schematic` provides a pure, deterministic text visualization of the circuit -- one mode per line, in the circuit's own mode order, gates shown left to right in execution order -- which `draw` simply passes to `print`.

---


=== Literature
The component-level interpretation of beam splitters, phase rotations, squeezing, and loss is standard Gaussian/linear quantum optics, covered by the same background literature as Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005), plus specifically:

- #link("https://doi.org/10.1002/3527602976.ch13")[W. P. Schleich, *Quantum Optics in Phase Space*, chapter on optical interferometry (Wiley-VCH, 2001).]

`Circuit` and its serialization/schematic abstractions are software-design decisions of `catsy`; the references above support the physical meaning of the modeled optical components.

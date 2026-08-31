#import "@preview/physica:0.9.8"

// ==========================================
// CHAPTER 8
// ==========================================
= Chapter 8: Reusable Layouts and Circuit Schematics

Chapter 3 introduced `Circuit` as an *ordered gate sequence*. This chapter covers the parts of `Circuit` aimed specifically at building and inspecting reusable hardware-style layouts: the fluent builder API, repeated execution against different input states, save/load, and the plain-text schematic renderer. A `Circuit` is the execution/orchestration abstraction; there is no separate optical-bench or component hierarchy. The public state operations remain ordinary transformations, while a `Gate` binds a *circuit-compatible transform* to its modes and parameters.

== Gates in a circuit

There is no separate component abstraction. A `Gate` already contains everything needed to describe one concrete optical transformation:

```python
Gate(
    name="BeamSplitter",
    transform=<circuit transform for beam_splitter>,
    modes=("a", "b"),
    kwargs={"eta": 0.5},
)
```

The `name` is the noun used as the stable gate identity and serialization key. The `transform` is the executable mathematical operation with the `GateTransform` calling convention `(state, modes, **kwargs)`; `modes` identify where it acts; and `kwargs` contain the parameters bound to this particular Gate instance. This circuit-transform interface is distinct from the direct state-operation signatures exposed by `catsy.operations`.

For normal application code, prefer the fluent `Circuit` methods. Raw `Gate` construction is mainly useful when working with the circuit registry or serialization layer; the fluent builder automatically supplies the correct circuit transform.

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

Each convenience method constructs one fully bound `Gate` and appends it to the circuit -- see Chapter 3 for the distinction between the public state-operation API and the circuit `GateTransform` registry. The same Gate can also be constructed explicitly when the appropriate registered circuit transform is supplied to `transform`.

== Execution

A `Circuit` can be run repeatedly against different input states without rebuilding its gate sequence -- `run` never mutates `self._gates`:

```python
first = circuit.run(input_state_a)
second = circuit.run(input_state_b)
```

An empty circuit (no gates) simply returns whatever state it's given unchanged, rather than raising -- `run`'s only hard requirements are that the circuit has at least one registered mode, and that it ends up with a state from somewhere (either a passed-in `initial_state`, or its own `InitialState` gate).

== Serialization and text schematic

`Circuit` is JSON-persistent via `to_dict`/`from_dict`/`save`/`load` (Chapter 3). Serialization records the Gate's stable name, target modes, and bound parameters rather than Python function objects. `render_schematic` provides a pure, deterministic text visualization of the circuit -- one mode per line, in the circuit's own mode order, gates shown left to right in execution order -- which `draw` simply passes to `print`.

The resulting separation is intentional: a public operation describes *what transformation means mathematically*; a Gate describes *one bound use of a circuit-compatible transform*; a Circuit describes *when and in what order those bound transformations execute*.

---


=== Literature
The component-level interpretation of beam splitters, phase rotations, squeezing, and loss is standard Gaussian/linear quantum optics, covered by the same background literature as Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005), plus specifically:

- #link("https://doi.org/10.1002/3527602976.ch13")[W. P. Schleich, *Quantum Optics in Phase Space*, chapter on optical interferometry (Wiley-VCH, 2001).]

`Circuit`, `Gate`, `Mode`, and their serialization/schematic abstractions are software-design decisions of `catsy`; the references above support the physical meaning of the modeled optical transformations.

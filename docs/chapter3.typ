#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 3
// ==========================================
= Chapter 3: Circuit Architecture & Gate Serialization

The `Circuit` class is the generic imperative sequencing layer of the toolkit. It stores an ordered sequence of fully bound `Gate` instances over named modes. A `Gate` contains the stable noun-style gate name, its executable `transform`, the target `modes`, and the bound `kwargs`.

== Gate instances and direct execution
The central abstraction is deliberately small:

```python
@dataclass(frozen=True)
class Gate:
    name: str
    transform: GateTransform
    modes: tuple[str, ...]
    kwargs: dict[str, object]

    def apply(self, state):
        return self.transform(state, self.modes, **self.kwargs)
```

The mathematical transformations remain ordinary verb-named functions such as `squeeze`, `rotate`, `displace`, `beam_splitter`, `loss`, and `thermal_loss`. A concrete Gate instance binds one such transform to its noun-style identity and parameters:

```python
squeezer = Gate(
    name="Squeezer",
    transform=squeeze,
    modes=("a",),
    kwargs={"r": 0.5},
)
```

The same Gate can be applied directly with `squeezer.apply(state)` or attached to a circuit with `circuit.add_gate(squeezer)`. Construction and attachment never execute the transformation.

== Fluent circuit construction
Registered transforms are also exposed as fluent Circuit methods. These methods construct the same kind of Gate instance and append it to the circuit:

```python
circuit.squeeze("a", r=0.5)
circuit.displace("a", alpha=0.2)
```

The fluent call is therefore equivalent to constructing a Gate explicitly and passing it to `add_gate`. The verb is the transformation/function name; the bound Gate carries the noun-style name.

== Serialization uses the Gate name
Function objects are intentionally kept out of JSON. A serialized gate contains its stable noun-style `name`, target modes, and primitive parameters. For example:

```json
{
  "gate": "BeamSplitter",
  "modes": ["a", "b"],
  "kwargs": {"eta": 0.5}
}
```

The registry maps serialized Gate names back to their transform functions when loading a circuit. Execution itself uses the transform already stored in each Gate instance.

== Sequential execution
The `run` method validates the Gate modes and then applies the bound Gates in order:

```python
for gate in self._gates:
    current_state = gate.apply(current_state)
```

This gives a linear execution path: `Gate` → `transform` → state. The circuit does not need a separate operation record or execution wrapper.

== State storage and roundtrip guarantee
Circuits remain persistent through `save` and `load`. Serialization stores only the Gate name, modes, and bound parameters; deserialization reconstructs a concrete Gate with the corresponding transform. A circuit can also contain an `InitialState` Gate. When present, `run()` can construct the initial state from that Gate; an explicit state passed to `run()` overrides it.


=== Literature
This chapter is primarily software-architectural rather than a derivation of new physics. The physical meaning of the registered gates and their sequential composition follows the same Gaussian-circuit formalism cited in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005). The registry, serialization, and validation mechanisms described here are implementation choices of `catsy`, not claims of a unique physical formalism.

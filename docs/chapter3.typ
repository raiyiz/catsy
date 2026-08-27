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

== Modes are owned, not just named
A `Circuit`'s modes are registered through `circuit.mode(name)`, which returns a small `Mode` object rather than the plain name:

```python
@dataclass(frozen=True, eq=False)
class Mode:
    name: str
    owner: Circuit | None = None
```

The returned `Mode` is *owned* by the circuit that produced it (`owner is circuit`). `circuit.add_mode(name)` is a thin fluent wrapper around `mode()` that discards the handle, for callers who only need the name registered. A `Mode` with `owner=None` is *free*: not tied to any circuit, e.g. for a one-off Gate built without a `Circuit` at all.

Fluent gate-builder methods and `initial_state` accept either a plain, already-registered mode-name string or a `Mode` handle. Passing a `Mode` owned by a *different* circuit -- or a free one -- is rejected immediately, before a `Gate` is even constructed:

```python
a = circuit_1.mode("a")
circuit_2.squeeze(a, r=0.5)  # ValueError: belongs to circuit_1, not circuit_2
```

This is what stops a mode meant for one circuit from silently ending up wired into a different one.

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

A complex Python number is not JSON-serializable, but the Displacer gate and `InitialState(kind="coherent")` both accept a complex displacement amplitude `alpha` as a convenience. Rather than reject it at serialization time, `Gate.kwargs` canonicalizes `alpha` into its real-valued quadratures `x`, `p` (via `GaussianState._normalize_translation`,
$x = sqrt(2) Re(alpha), quad p = sqrt(2) Im(alpha)$) *at gate-construction time* -- before a `Gate` is even built, not only when it later runs:

```python
circuit.displace("a", alpha=0.2 + 0.1j)
circuit.to_dict()["gates"][-1]["kwargs"]
# -> {"x": 0.28284..., "p": 0.14142...}  -- never {"alpha": ...}
```

This is a lossless, format-only substitution: both transforms accept `x`/`p` directly as an equivalent input, so nothing about the physics changes -- only what's stored on disk. It means a circuit built with `alpha=` round-trips through `save`/`load` and through `SimulationJournal` (Chapter 9) the same as one built with `x=`/`p=` directly.

== Sequential execution
The `run` method applies the bound Gates in order:

```python
for gate in self._gates:
    current_state = gate.apply(current_state)
```

This gives a linear execution path: `Gate` → `transform` → state. The circuit does not need a separate operation record or execution wrapper. Mode registration is validated once, in `add_gate` -- not re-checked here.

== State storage and roundtrip guarantee
Circuits remain persistent through `save` and `load`. Serialization stores only the Gate name, modes, and bound parameters; deserialization reconstructs a concrete Gate with the corresponding transform. A circuit can also contain an `InitialState` Gate. When present, `run()` can construct the initial state from that Gate; an explicit state passed to `run()` overrides it.


=== Literature
This chapter is primarily software-architectural rather than a derivation of new physics. The physical meaning of the registered gates and their sequential composition follows the same Gaussian-circuit formalism cited in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005). The registry, serialization, and validation mechanisms described here are implementation choices of `catsy`, not claims of a unique physical formalism.

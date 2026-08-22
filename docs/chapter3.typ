#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 3
// ==========================================
= Chapter 3: Circuit Architecture & Gate Serialization

The `Circuit` class is the generic imperative sequencing layer of the toolkit. It stores an ordered sequence of executable callables over named modes and does not know whether a gate is Gaussian, Fock-space, or belongs to another backend. The state and gate callable define the computational domain; the circuit only sequences and executes them.

== Callable gates and direct execution
The circuit stores the executable gate itself rather than an intermediate gate record. Gates are typed by the generic `Gate` protocol, while the circuit remains backend-agnostic:

```python
class Gate(Protocol):
    name: str
    def __call__(self, state, modes, **kwargs): ...
```

The built-in gates are plain function objects such as `squeeze`, `rotate`, `displace`, `beam_splitter`, `loss`, and `thermal_loss`. Each gate carries an explicit `name` attribute as part of the Catsy gate contract. Higher-level domain objects can therefore attach a callable directly to a `Circuit`. There is no runtime string-to-function dispatch step.

== Serialization uses the function name
Function objects are intentionally kept out of JSON. When a circuit is serialized, each gate stores only the callable's explicit `name` attribute, together with its target modes and primitive parameters. For example:

```json
{
  "gate": "beam_splitter",
  "modes": ["a", "b"],
  "kwargs": {"eta": 0.5}
}
```

The small deserialization mapping is used only when loading JSON; execution never consults it. This preserves the direct callable contract at runtime while keeping the file format simple. Custom callables can be registered for deserialization with `Circuit.register`; the circuit still executes the callable object that was attached to it.

== Compilation and sequential execution
The `run` method resolves the initial state, validates that every gate targets a registered mode, and then invokes the stored callable directly:

```python
for idx, (gate, modes, kwargs) in enumerate(self._gates):
    for mode in modes:
        if mode not in self.modes:
            raise ValueError(...)
    current_state = gate(current_state, modes, **kwargs)
```

This makes the execution path linear: gate callable → `Circuit` → state. The circuit does not need to understand the domain-specific origin of an operation.

== State storage and roundtrip guarantee
Circuits remain persistent through `save` and `load`. The runtime callable is replaced in the JSON representation by its bare function name, and the loader resolves that name back to a callable before attaching it to the reconstructed circuit. A circuit can also contain an `initial_state` gate. When present, `run()` can construct the initial state from that gate; an explicit state passed to `run()` overrides it.


=== Literature
This chapter is primarily software-architectural rather than a derivation of new physics. The physical meaning of the registered gates and their sequential composition follows the same Gaussian-circuit formalism cited in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005). The registry, serialization, and validation mechanisms described here are implementation choices of `catsy`, not claims of a unique physical formalism.

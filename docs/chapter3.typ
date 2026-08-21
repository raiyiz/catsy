#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 3
// ==========================================
= Chapter 3: Compiler Architecture & Extensible Gate Registry

The `GaussianCircuit` class acts as the imperative sequencing layer of the toolkit. At the physics level, the circuit is a finite composition of Gaussian channels and unitaries; the resulting mathematical evolution remains within the Gaussian formalism described by #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)]. Its software design strictly separates the definition of the algorithmic gate sequence from the mathematical execution layer (*execution engine*). This keeps the system fully extensible without needing to modify the compiler core itself.

== Callable operations and direct execution
The circuit stores the executable operation itself rather than an intermediate `CircuitOperation` record. A Gaussian operation is a callable satisfying the `GaussianOperation` protocol:

```python
class GaussianOperation(Protocol):
    def __call__(
        self, state: GaussianState, modes: Modes, **kwargs: ParameterValue
    ) -> GaussianState: ...
```

The built-in operations are plain functions such as `squeeze`, `rotate`, `displace`, `beam_splitter`, `loss`, and `thermal_loss`. Higher-level domain objects can therefore attach a callable directly to a `GaussianCircuit`. There is no runtime string-to-function dispatch step.

== Serialization uses the function name
Function objects are intentionally kept out of JSON. When a circuit is serialized, each operation stores only the callable's bare Python `__name__`, together with its target modes and primitive parameters. For example:

```json
{
  "name": "beam_splitter",
  "modes": ["a", "b"],
  "kwargs": {"eta": 0.5}
}
```

The small deserialization mapping is used only when loading JSON; execution never consults it. This preserves the direct callable contract at runtime while keeping the file format simple. Custom callables can be registered for deserialization with `GaussianCircuit.register`; the circuit still executes the callable object that was attached to it.

== Compilation and sequential execution
The `compile_and_run` method resolves the initial state, validates that every operation targets a registered mode, and then invokes the stored callable directly:

```python
for idx, (op, modes, kwargs) in enumerate(self._operations):
    for mode in modes:
        if mode not in self.modes:
            raise ValueError(...)
    current_state = op(current_state, modes, **kwargs)
```

This makes the execution path linear: callable operation -> `GaussianCircuit` -> `GaussianState`. The circuit does not need to understand the domain-specific origin of an operation.

== State storage and roundtrip guarantee
Circuits remain persistent through `save` and `load`. The runtime callable is replaced in the JSON representation by its bare function name, and the loader resolves that name back to a callable before attaching it to the reconstructed circuit. The per-mode initial amplitudes set via `add_mode(..., alpha=...)` round-trip alongside the operation list.


=== Literature
This chapter is primarily software-architectural rather than a derivation of new physics. The physical meaning of the registered gates and their sequential composition follows the same Gaussian-circuit formalism cited in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005). The registry, serialization, and validation mechanisms described here are implementation choices of `catsy`, not claims of a unique physical formalism.

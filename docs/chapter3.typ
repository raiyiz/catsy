#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 3
// ==========================================
= Chapter 3: Compiler Architecture & Extensible Gate Registry

The `GaussianCircuit` class acts as the imperative sequencing layer of the toolkit. Its software design strictly separates the definition of the algorithmic gate sequence from the mathematical execution layer (*execution engine*). This keeps the system fully extensible without needing to modify the compiler core itself.

== The data-driven operations model (`CircuitOperation`)
To guarantee straightforward serializability of the quantum circuit, the compiler does not store operations as direct function references, but decouples them into flat data structures via the `CircuitOperation` class.

```python
@dataclass(frozen=True)
class CircuitOperation:
    name: str
    modes: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
```

This structure keeps every circuit inherently serializable (`JSON`), since a gate is fully described by its logical registration key (`name`), its addressing modes (`modes`), and its primitive parameters (`kwargs`).

== The open-closed pattern (`OPERATION_REGISTRY`)
The decoupling between gate invocation and mathematical backend is realized via a global dispatch dictionary (`OPERATION_REGISTRY`). The compiler dynamically consults this mapping at runtime. Functional extension happens through the class method `register`:

```python
class CircuitOpCallable(Protocol):
    def __call__(self, state: GaussianState, modes: tuple[str, ...], **kwargs: Any) -> GaussianState: ...

OPERATION_REGISTRY: dict[str, CircuitOpCallable] = {
    "Squeezing": lambda s, m, **kw: GaussianOperations.apply_squeezing(s, m, **kw),
    "PhaseRotation": lambda s, m, **kw: GaussianOperations.apply_phase_rotation(s, m, **kw),
    "BeamSplitter": lambda s, m, **kw: GaussianOperations.apply_beam_splitter(s, m, m, **kw),
    "Loss": lambda s, m, **kw: GaussianOperations.apply_loss(s, m, **kw),
    "ThermalLossChannel": lambda s, m, **kw: QBSChannels.thermal_loss(m, **kw).apply(s),
}
```

A new gate type (e.g. a custom error or hardware model) can be injected at runtime via `GaussianCircuit.register("MyCustomOp", fn)`. Static type checkers (MyPy/Pyright) then enforce the correct functional interface signature `(GaussianState, tuple[str, ...], **kwargs) -> GaussianState` via the `CircuitOpCallable` protocol.

== Compilation and sequential execution
The `compile_and_run` method turns the abstract gate chain into a concrete phase-space evolution. Before the actual computation, the compiler performs a two-stage validation:
1. *Mode validation:* checks that every gate's target modes are registered in the circuit.
2. *Vacuum initialization:* if no `initial_state` is given, an exact multi-mode vacuum state $V_0 = 1/2 I_(2n)$ is generated.

The sequential computation loop is implemented in the source as follows:

```python
def compile_and_run(self, initial_state: GaussianState | None = None) -> GaussianState:
    if not self.modes:
        raise ValueError("Circuit has no registered modes.")
    
    current_state = GaussianOperations.create_vacuum(self.modes) if initial_state is None else initial_state
    if set(current_state.modes) != set(self.modes):
        raise ValueError("Initial state modes mismatch circuit modes.")

    for idx, op in enumerate(self._operations):
        for m in op.modes:
            if m not in self.modes:
                raise ValueError(f"Op #{idx} ({op.name}): mode '{m}' is not registered.")
        if op.name not in OPERATION_REGISTRY:
            raise KeyError(f"Unknown operation '{op.name}'.")
        
        # Dynamic dispatch to the mathematical backend
        current_state = OPERATION_REGISTRY[op.name](current_state, op.modes, **op.kwargs)
    return current_state
```

== State storage and roundtrip guarantee
Circuits can be made persistent directly on the filesystem via the native `save` and `load` methods. The interplay of ordered lists and primitive data types in `to_dict` ensures that loaded circuits generate mathematically identical covariance matrices:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "modes": list(self.modes),
        "operations": [
            {"name": op.name, "modes": list(op.modes), "kwargs": op.kwargs}
            for op in self._operations
        ]
    }
```

---


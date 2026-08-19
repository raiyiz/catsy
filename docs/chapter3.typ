#import "@preview/physica:0.9.8": *

// ==========================================
// CHAPTER 3
// ==========================================
= Chapter 3: Compiler Architecture & Extensible Gate Registry

The `GaussianCircuit` class acts as the imperative sequencing layer of the toolkit. At the physics level, the circuit is a finite composition of Gaussian channels and unitaries; the resulting mathematical evolution remains within the Gaussian formalism described by #link("https://doi.org/10.1103/RevModPhys.84.621")[Weedbrook et al. (2012)]. Its software design strictly separates the definition of the algorithmic gate sequence from the mathematical execution layer (*execution engine*). This keeps the system fully extensible without needing to modify the compiler core itself.

== The data-driven operations model (`CircuitOperation`)
To guarantee straightforward serializability of the quantum circuit, the compiler does not store operations as direct function references, but decouples them into flat data structures via the `CircuitOperation` class.

```python
@dataclass
class CircuitOperation:
    """One step in a compiled circuit: a registry key + its target modes and
    kwargs. Deliberately holds no function reference, so it stays trivially
    JSON-serializable."""
    name: str
    modes: tuple[str, ...]
    kwargs: dict[str, int | float | complex]
```

This structure keeps every circuit inherently serializable (`JSON`), since a gate is fully described by its logical registration key (`name`), its addressing modes (`modes`), and its primitive parameters (`kwargs`).

== The open-closed pattern (`OPERATION_REGISTRY`)
The decoupling between gate invocation and mathematical backend is realized via a global dispatch dictionary (`OPERATION_REGISTRY`). The compiler dynamically consults this mapping at runtime. Functional extension happens through the class method `register`:

```python
class GaussianOperation(Protocol):
    def __call__(
        self, state: GaussianState, modes: Modes, **kwargs: ParameterValue
    ) -> GaussianState: ...

def _op_squeeze(state, modes, **kwargs):
    return state.squeeze(mode=modes[0], r=kwargs["r"], theta=kwargs["theta"])

def _op_rotate(state, modes, **kwargs):
    return state.rotate(mode=modes[0], phi=kwargs["phi"])

def _op_displace(state, modes, **kwargs):
    return state.displace(mode=modes[0], x=kwargs["x"], p=kwargs["p"])

def _op_beam_splitter(state, modes, **kwargs):
    return state.beam_splitter(mode_a=modes[0], mode_b=modes[1], eta=kwargs["eta"])

def _op_loss(state, modes, **kwargs):
    return state.loss(mode=modes[0], eta=kwargs["eta"])

def _op_thermal_loss(state, modes, **kwargs):
    channel = LossChannels.thermal_loss(mode=modes[0], eta=kwargs["eta"],
                                         n_thermal=kwargs["n_thermal"])
    return channel.apply(state)

OPERATION_REGISTRY: dict[str, GaussianOperation] = {
    "Squeezing": _op_squeeze,
    "PhaseRotation": _op_rotate,
    "Displacement": _op_displace,
    "BeamSplitter": _op_beam_splitter,
    "Loss": _op_loss,
    "ThermalLossChannel": _op_thermal_loss,
}
```

Each registered callable is a small, explicit wrapper rather than a generic `**kwargs`-forwarding lambda, so a malformed or missing parameter raises a clear `KeyError` naming the missing kwarg instead of an opaque `TypeError` from deep inside `GaussianState`. Note the channel factory here is `LossChannels` (Chapter 2) -- there is no separate "QBS" channel class. A new gate type (e.g. a custom error or hardware model) can be injected at runtime via `GaussianCircuit.register("MyCustomOp", fn)`. Static type checkers (MyPy/Pyright) then enforce the correct functional interface signature `(GaussianState, Modes, **kwargs: ParameterValue) -> GaussianState` via the `GaussianOperation` protocol.

== Compilation and sequential execution
The `compile_and_run` method turns the abstract gate chain into a concrete phase-space evolution. Before the actual computation, the compiler performs a two-stage initialization/validation:
1. *Initial-state resolution:* if no `initial_state` is given, each registered mode starts either in vacuum or, if `add_mode` was called with an explicit `alpha`, in the corresponding coherent state $ket(alpha)$ -- so a circuit can describe its own non-vacuum input without the caller having to build one separately. If an `initial_state` *is* given, its modes must match the circuit's registered modes as a set, and it is reordered (via `reorder_modes`, Chapter 5) into the circuit's canonical mode order once at this boundary, so every subsequent operation sees a consistent positional layout regardless of the order the caller's state happened to use.
2. *Mode validation:* checks that every gate's target modes are registered in the circuit.

The sequential computation loop is implemented in the source as follows:

```python
def compile_and_run(self, initial_state: GaussianState | None = None) -> GaussianState:
    if not self.modes:
        raise ValueError("Circuit has no registered modes.")

    if initial_state is None:
        alphas = [self._initial_alphas.get(m, 0.0) for m in self.modes]
        current_state = GaussianState.coherent(self.modes, alphas)
    else:
        if set(initial_state.modes) != set(self.modes):
            raise ValueError("Initial state's modes don't match the circuit's modes.")
        current_state = initial_state.reorder_modes(self.modes)

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
        # Stored as [re, im] pairs -- complex isn't JSON-serializable.
        "initial_alphas": {
            m: [float(np.real(a)), float(np.imag(a))]
            for m, a in self._initial_alphas.items()
        },
        "operations": [
            {"name": op.name, "modes": list(op.modes), "kwargs": op.kwargs}
            for op in self._operations
        ]
    }
```

The per-mode initial amplitudes set via `add_mode(..., alpha=...)` round-trip through `initial_alphas` alongside the operation list, so a saved-and-reloaded circuit reproduces the exact same default input state, not just the same gate sequence.

---



=== Literature
This chapter is primarily software-architectural rather than a derivation of new physics. The physical meaning of the registered gates and their sequential composition follows the same Gaussian-circuit formalism cited in Chapter 1 (Weedbrook et al. 2012; Braunstein and van Loock 2005). The registry, serialization, and validation mechanisms described here are implementation choices of `catsy`, not claims of a unique physical formalism.

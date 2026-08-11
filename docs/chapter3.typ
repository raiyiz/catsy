
// ==========================================
// KAPITEL 3
// ==========================================
= Kapitel 3: Compiler-Architektur & Extensible Gate Registry

Die Klasse `GaussianCircuit` fungiert als imperative Sequenzierungsschicht des Toolkits. Ihr Software-Design trennt die Definition der algorithmischen Gatter-Abfolge strikt von der mathematischen Ausführungsebene (*Execution Engine*). Dadurch bleibt das System vollständig erweiterbar, ohne den eigentlichen Compiler-Kern modifizieren zu müssen.

== Das datengetriebene Operations-Modell (`CircuitOperation`)
Um eine einfache Serialisierbarkeit des Quantenschaltkreises zu garantieren, speichert der Compiler Operationen nicht als direkte Funktionsreferenzen, sondern entkoppelt als flache Datenstrukturen in Form der Klasse `CircuitOperation`.

```python
@dataclass(frozen=True)
class CircuitOperation:
    name: str
    modes: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
```

Durch diese Struktur bleibt jeder Schaltkreis inhärent serialisierbar (`JSON`), da ein Gatter vollständig durch seinen logischen Registrierungsschlüssel (`name`), seine Adressierungs-Moden (`modes`) und seine primitiven Parameter (`kwargs`) beschrieben wird.

== Das Open-Closed-Muster (`OPERATION_REGISTRY`)
Die Entkopplung zwischen Gatter-Aufruf und mathematischem Backend wird über ein globales Dispatch-Dictionary (`OPERATION_REGISTRY`) realisiert. Der Compiler greift zur Laufzeit dynamisch auf diese Zuordnung zu. Eine funktionale Erweiterung erfolgt über die Klassenmethode `register`:

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

Ein neuer Gatter-Typ (z. B. ein benutzerdefiniertes Fehler- oder Hardware-Modell) lässt sich über `GaussianCircuit.register("MyCustomOp", fn)` zur Laufzeit injizieren. Statische Type-Checker (MyPy/Pyright) erzwingen dabei über das `CircuitOpCallable`-Protokoll die korrekte funktionale Schnittstellensignatur `(GaussianState, tuple[str, ...], **kwargs) -> GaussianState`.

== Kompilation und sequentielle Ausführung
Die Methode `compile_and_run` überführt die abstrakte Gatter-Kette in eine konkrete Phasenraum-Evolution. Vor der eigentlichen Berechnung führt der Compiler eine zweistufige Validierung durch:
1. *Moden-Validierung:* Prüft, ob alle Gatter-Zielmoden im Schaltkreis registriert sind.
2. *Vakuum-Initialisierung:* Ist kein `initial_state` gegeben, wird ein exakter Multi-Moden-Vakuumzustand $V_0 = 1/2 I_(2n)$ erzeugt.

Die sequentielle Berechnungsschleife ist im Quelltext wie folgt implementiert:

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
        
        # Dynamischer Dispatch an das mathematische Backend
        current_state = OPERATION_REGISTRY[op.name](current_state, op.modes, **op.kwargs)
    return current_state
```

== Zustandsspeicherung und Roundtrip-Garantie
Schaltkreise können direkt über die nativen Methoden `save` und `load` auf Dateisystem-Ebene persistent gemacht werden. Das Zusammenspiel aus geordneten Listen und primitiven Datentypen in `to_dict` stellt sicher, dass geladene Schaltkreise mathematisch identische Kovarianzmatrizen generieren:

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

#import "@preview/physica:0.9.8": *

= Kapitel 1: Mathematisches Fundament & Phasenraum-Darstellung

Das vorliegende Framework ist für die hocheffiziente Simulation von kontinuierlichen Variablen (CV) in der Quantenoptik optimiert. Im Gegensatz zur expliziten Darstellung im unendlich-dimensionalen Fock-Raum nutzt der Gaußsche Phasenraum-Layer eine exakte Parametrisierung über die ersten und zweiten statistischen Momente.

== Symplektische Konventionen
Für ein System aus n optischen Moden definieren wir den Vektor der hermiteschen Quadraturoperatoren als:
$r = "matrix"(q_1, p_1, q_2, p_2, dots, q_n, p_n)^T$

Die Operatoren erfüllen die kanonischen Kommutationsrelationen (CCR), welche symplektisch ausgedrückt werden als:
$[r_i, r_j] = i Omega_(i j)$

wobei Omega die fundamentale symplektische Form darstellt. Das Toolkit implementiert Omega als orthogonale Summe (bigoplus) von 2 times 2 Blöcken über die gesamte Modenanzahl:
 Omega = col.big(oplus, k=1, n) pmatrix(0, 1; -1, 0) 

Unter Verwendung der Konvention hbar = 1 ist das Schrotrauschen-Limit (*Shot-Noise Limit*) des quantenmechanischen Vakuums definiert durch die Kovarianzmatrix:
$V_0 = 1/2 I_(2n)$

== Mathematische Zustandsspezifikation (`GaussianState`)
Ein quantenmechanischer Zustand rho ist im Phasenraum vollständig durch seinen Verschiebungsvektor d und seine Kovarianzmatrix V charakterisiert, sofern die Wigner-Funktion eine Gauß-Verteilung aufweist.

1. *Verschiebungsvektor ($d in R^(2n)$):*
   $d_i = expval(r_i)_rho = text(tr)(rho r_i)$

2. *Kovarianzmatrix ($V in R^(2n times 2n)$):*
   $V_(i j) = 1/2 expval(\{r_i - d_i, r_j - d_j\})_rho$
   Aus der Positivität der Dichtematrix rho >= 0 folgt direkt die Unschärferelation im Phasenraum in Form der Robertson-Schrödinger-Ungleichung:
    V + i/2 Omega >= 0 

3. *Zustandspolarität und Reinheit (Purity):*
   Die Reinheit mu = text(Tr)(rho²) berechnet sich im Phasenraum direkt aus der Determinante von V:
   $mu = 1 / (2^n sqrt(det(V)))$
   Für einen reinen Zustand gilt strikt $det(V) = (1/2)^(2n) = 1/(4^n)$, womit mu = 1 erreicht wird.

== Code-Architektur & Validierung von `GaussianState`

Die Python-Klasse `GaussianState` spiegelt diese Invarianten eins zu eins wider. Sie erzwingt strenge Validierungsregeln während der Initialisierung, um unphysikalische Zustände zur Laufzeit auszuschließen.

```python
@dataclass
class GaussianState:
    """A multi-mode Gaussian state, fully described by (modes, d, V)."""
    modes: tuple[str, ...]
    displacement: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        n_modes = len(self.modes)
        if len(set(self.modes)) != n_modes:
            raise ValueError(f"Duplicate mode names in {self.modes!r}.")
        
        expected_dim = 2 * n_modes
        if self.displacement.shape != (expected_dim,):
            raise ValueError(
                f"displacement must have shape ({expected_dim},), "
                f"got {self.displacement.shape}."
            )
        if self.covariance.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"covariance must have shape ({expected_dim}, {expected_dim}), "
                f"got {self.covariance.shape}."
            )
```

Die Dimensionen werden streng auf $2n$ überwacht. Ein Fehler in den Eingabedaten führt sofort zu einem `ValueError`, noch bevor nachfolgende unitäre Transformationen fehlerhafte Matrixberechnungen triggern können.

---

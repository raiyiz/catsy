#import "@preview/physica:0.9.8": *

= Kapitel 2: Gaußsche unitäre Transformationen & Kanäle

Gaußsche Operationen sind dadurch definiert, dass sie die Gauß-Struktur der Wigner-Funktion invariant halten. Mathematisch entsprechen unitäre Operatoren dieser Klasse affinen symplektischen Transformationen im Phasenraum. Nicht-unitäre Prozesse (Dekohärenz und Rauschen) werden als CPTP-Abbildungen (*Completely Positive Trace-Preserving Maps*) über Gaußsche Kanäle modelliert.

== Unitäre Gatter-Transformationen (`GaussianOperations`)
Jede rein Gaußsche unitäre Transformation $hat(U)$ induziert eine lineare Transformation im Phasenraum, die sich über eine symplektische Matrix $S in S_p(2n, RR)$ ausdrückt. Für den Verschiebungsvektor $d$ und die Kovarianzmatrix $V$ gilt:
$d -> S d$
$V -> S V S^T$

Die Erhaltung der kanonischen Kommutationsrelationen erfordert strikt, dass $S$ die symplektische Form bewahrt:
$S Omega S^T = Omega$

Das Toolkit implementiert drei fundamentale Basistransformationen in `GaussianOperations`:

1. *Squeezing-Operator ($hat(S)_k (r, theta)$):*
   Der lokale Quetsch-Operator auf der Mode $k$ reduziert die Varianz in einer Quadratur unter das Schrotrauschen-Limit, während die konjugierte Quadratur verstärkt wird. Die lokale symplektische Matrix lautet:
   $ S_("local") = mat(cos(theta), -sin(theta); sin(theta), cos(theta)) mat(e^(-r), 0; 0, e^r) mat(cos(theta), sin(theta); -sin(theta), cos(theta)) $

2. *Phasenrotation ($hat(R)_k (phi)$):*
   Eine passive, energieerhaltende Transformation, die den Phasenraum um den Winkel $phi$ dreht:
   $ S_("local") = mat(cos(phi), -sin(phi); sin(phi), cos(phi)) $

3. *Strahlteiler ($hat(B)_(k, m)(eta)$):*
   Ein verlustfreier Strahlteiler koppelt die Moden $k$ und $m$ mit einer Leistungstransmissivität $eta$. Er erzeugt Verschränkung zwischen unabhängigen Moden:
   $t = sqrt(eta), quad r_c = sqrt(1 - eta)$
   Die Transformation mischt die Modenpaare gemäss:
   $mat(d_k ; d_m) -> mat(t I_2, r_c I_2; -r_c I_2, t I_2) mat(d_k ; d_m)$

== Allgemeine Gaußsche Kanäle (`GaussianChannel`)
Rauschen und Dissipation können nicht mehr rein über unitäre $S$-Matrizen abgebildet werden. Ein allgemeiner Gaußscher Kanal wird mathematisch durch zwei reelle Matrizen $X$ und $Y$ beschrieben:
$d -> X d + d_0$
$V -> X V X^T + Y$

Damit diese Transformation eine physikalische CPTP-Abbildung darstellt, muss die Matrix $Y$ die Rausch-Ungleichung erfüllen:
$Y + i/2 Omega - i/2 X Omega X^T >= 0$

Das Toolkit implementiert diese Abbildungen hocheffizient über eine globale Koordinaten-Einbettung in der Klasse `GaussianChannel`:

```python
@dataclass(frozen=True)
class GaussianChannel:
    """A general Gaussian channel d' = X@d + d0, V' = X@V@X.T + Y acting on a subset of modes."""
    target_modes: tuple[str, ...]
    X: np.ndarray = field(hash=False)
    Y: np.ndarray = field(hash=False)
    d0: np.ndarray = field(hash=False)

    def apply(self, state: GaussianState) -> GaussianState:
        global_dim = len(state.displacement)
        X_global = np.eye(global_dim)
        Y_global = np.zeros((global_dim, global_dim))
        d0_global = np.zeros(global_dim)

        # Lokale Matrizen in die globalen Indizes der Moden einbetten
        for l_idx1, m1 in enumerate(self.target_modes):
            gi1 = state.get_mode_index(m1)
            d0_global[gi1 : gi1 + 2] = self.d0[l_idx1 * 2 : l_idx1 * 2 + 2]
            for l_idx2, m2 in enumerate(self.target_modes):
                gi2 = state.get_mode_index(m2)
                X_global[gi1 : gi1 + 2, gi2 : gi2 + 2] = self.X[l_idx1 * 2 : l_idx1 * 2 + 2, l_idx2 * 2 : l_idx2 * 2 + 2]
                Y_global[gi1 : gi1 + 2, gi2 : gi2 + 2] = self.Y[l_idx1 * 2 : l_idx1 * 2 + 2, l_idx2 * 2 : l_idx2 * 2 + 2]

        new_d = X_global @ state.displacement + d0_global
        new_V = X_global @ state.covariance @ X_global.T + Y_global
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)
```

== Typische Optische Rauschkanäle (`QBSChannels`)
Das Framework stellt über das Factory-Objekt `QBSChannels` physikalische Standardkanäle bereit:

1. *Thermischer Verlustkanal (`thermal_loss`):*
   Modelliert die Kopplung an ein thermisches Bad mit mittlerer Photonenzahl $n_("thermal")$. Mit der Dämpfung $eta$ gilt:
   $ X = sqrt(eta) I_2, quad Y = (1 - eta)(n_("thermal") + 1/2) I_2 $

2. *Klassischer Phasen-Jitter (`classical_phase_jitter`):*
   Simuliert eine stochastische Phasenfluktuation in Kleinwinkelnäherung. Dies fügt Rauschen *ausschließlich* der Impulsquadratur $p$ hinzu, während die Ortsquadratur $q$ perfekt erhalten bleibt:
   $ X = I_2, quad Y = mat(0, 0; 0, sigma_phi^2) $

---


// ==========================================
// KAPITEL 4
// ==========================================
= Kapitel 4: Quanten-Messprozesse & Zustandskonditionierung

Messungen an kontinuierlichen Variablen induzieren eine nicht-glorifizierte, irreversible Projektion des quantenmechanischen Gesamtzustands. Da das Toolkit im Gaußschen Phasenraum-Layer operiert, muss die mathematische Beschreibung der Wellenfunktionsreduktion (*Wigner-Kollaps*) ohne unendlich-dimensionale Projektionsoperatoren auskommen. Das Framework löst dies über die exakte Implementierung des symplektischen Schur-Komplements in der Klasse `GaussianMeasurements`.

== Homodyne Messung (`homodyne_measurement`)
Die homodyne Detektion misst eine frei wählbare Linearkombination aus den kanonischen Quadraturoperatoren q und p einer Zielmode. Dieser Prozess ist intrinsisch stochastisch: Das System kollabiert auf den Eigenzustand des Quadratuperators, und die restlichen Moden werden *konditioniert* (verschränkte Subsysteme passen ihre Statistik augenblicklich an).

=== Mathematische Transformation & Moden-Rotation
Um die Messung für einen beliebigen Lokaloszillator-Winkel phi (wobei phi=0 der Ortsquadratur q und phi=pi/2 der Impulsquadratur p entspricht) verallgemeinern zu können, transformiert der Code das System zuerst über eine globale passive Rotationsmatrix $R_("global")$ in die Eigenbasis des Messaufbaus. Dadurch reduziert sich jede homodyne Detektion mathematisch auf eine reine Messung der ersten Quadratur (q) der Zielmode.

Die Partitionierung der rotierten Kovarianzmatrix $V_("rot")$ erfolgt in vier strukturelle Submatrizen:
$ V_("rot") = mat(V_(M M), V_(M R); V_(R M), V_(R R)) $

- $V_(M M)$ (Skalar): Die inhärente Varianz der zu messenden Quadratur.
- $V_(M R)$ und $V_(R M)^T$ (Vektoren): Die Kreuzkorrelationen zwischen der Messmode und allen verbleibenden Systemmoden. Sie sind der mathematische Hebel, durch den die Verschränkung den Kollaps des Restsystems steuert.
- $V_(R R)$ (Matrix): Die isolierte Kovarianzmatrix der nicht-gemessenen Submoden.

=== Stochastisches Sampling & Schur-Konditionierung
Wird kein expliziter Messwert (`outcome`) vorgegeben, berechnet das Toolkit das physikalische Messergebnis stochastisch korrekt. Die Wahrscheinlichkeitsverteilung für das Ergebnis $x_m$ ist eine Gauß-Verteilung, zentriert um den rotierten Mittelwert der Quadratur mit der Breite der Quantenvarianz:
$x_m "sim" N(d_M, sqrt(V_(M M)))$

Der Kollaps der verbleibenden Moden wird anschließend über das *Schur-Komplement* berechnet. Die Evolution des Verschiebungsvektors $d_("cond")$ und der Kovarianzmatrix $V_("cond")$ ist im Code exakt wie folgt abgebildet:

```python
@staticmethod
def homodyne_measurement(
    state: GaussianState,
    measured_mode: str,
    phi: float,
    outcome: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, GaussianState]:
    n_modes = len(state.modes)
    idx_m = state.get_mode_index(measured_mode)

    # 1. Konstruktion des globalen Rotationsoperators für den LO-Winkel phi
    R_local = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    R_global = np.eye(2 * n_modes)
    R_global[idx_m : idx_m + 2, idx_m : idx_m + 2] = R_local

    # Rotation der Momente in die Messbasis
    d_rot = R_global @ state.displacement
    V_rot = R_global @ state.covariance @ R_global.T

    # Indizes aller nicht-gemessenen Quadraturen extrahieren
    idx_x = idx_m
    remaining_indices = [
        i for i in range(2 * n_modes) if i != idx_x and i != idx_m + 1
    ]

    # 2. Blockpartitionierung für das Schur-Komplement
    V_MM = V_rot[idx_x, idx_x]
    V_MR = V_rot[idx_x, remaining_indices]
    V_RM = V_rot[remaining_indices, idx_x]
    V_RR = V_rot[np.ix_(remaining_indices, remaining_indices)]

    # 3. Deterministisches oder stochastisches Sampling des Outcomes
    if outcome is None:
        rng = rng if rng is not None else np.random.default_rng()
        measured_value = rng.normal(loc=d_rot[idx_x], scale=np.sqrt(V_MM))
    else:
        measured_value = outcome

    # 4. Berechnung des bedingten Zustands (Wigner-Kollaps)
    # Die Verschiebung verschiebt sich proportional zur Abweichung vom Erwartungswert
    d_cond = d_rot[remaining_indices] + V_RM * (1.0 / V_MM) * (measured_value - d_rot[idx_x])
    # Die neue Kovarianzmatrix schrumpft; Unsicherheit wird durch Informationsextraktion reduziert
    V_cond = V_RR - np.outer(V_RM, V_MR) / V_MM

    remaining_modes = tuple(m for m in state.modes if m != measured_mode)
    return float(measured_value), GaussianState(remaining_modes, d_cond, V_cond)
```

=== Warum der Code das tut (Physikalische Kausalität)
- *`V_RM * (1.0 / V_MM) * (measured_value - d_rot[idx_x])`*: Wenn die gemessene Mode mit dem Restsystem verschränkt war (z. B. ein EPR-Paar), korreliert ihr Zustand mit den anderen Moden. Weicht das Messergebnis `measured_value` von seinem quantenmechanischen Mittelwert ab, zwingt diese Korrelation das verbleibende System zu einer makroskopischen Verschiebung im Phasenraum.
- *`- np.outer(V_RM, V_MR) / V_MM`*: Jede homodyne Messung extrahiert Information aus dem Gesamtsystem. Da die Kreuzkorrelationen $V_("RM")$ das Ausmaß des Quantenwissens über das Subsystem kodieren, subtrahiert das Schur-Komplement diese Unsicherheit exakt. Das verbleibende System schrumpft im Phasenraum entlang der verschränkten Achsen.

== Heterodyne Messung (`heterodyne_measurement`)
Die heterodyne (oder Doppel-Homodyn-) Detektion misst simultan beide konjugierten Quadraturen (q und p) einer Mode. Da $[q, p] = i != 0$, verbietet das Heisenberg'sche Unschärfeprinzip eine exakte gleichzeitige Messung ohne die Injektion von zusätzlichem Rauschen.

=== Das mathematische Vakuum-Port-Modell
Physikalisch entspricht die heterodyne Messung dem Splitten der Zielmode an einem 50:50-Strahlteiler, dessen zweiter Eingang mit einem unkorrelierten Vakuum-Zustand besetzt ist. Die beiden Ausgänge werden anschließend homodyn detektiert (einer misst q, der andere p). 

Dieses intrinsische Quantenrauschen wird im Code elegant und performant simuliert, ohne dass der Strahlteiler explizit im Phasenraum konstruiert werden muss: Das Vakuumrauschen wird direkt als additiver Term auf den Messblock aufgeschlagen:
$ V_("eff") = V_(M M) + 1/2 I_2 $

=== Implementierung & Rauschinjektion
Aufgrund dieser Rauschinjektion kollabiert die gemessene Mode nicht auf einen unendlich gequetschten Eigenzustand, sondern auf einen kohärenten Zustand (Projektion auf kohärente Zustände / *Husimi-Q-Repräsentation*).

```python
@staticmethod
def heterodyne_measurement(
    state: GaussianState,
    measured_mode: str,
    outcome: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, GaussianState]:
    idx_m = state.get_mode_index(measured_mode)
    dim = len(state.displacement)
    remaining_indices = [i for i in range(dim) if i < idx_m or i > idx_m + 1]

    # Partitionierung der reinen Systemkovarianz (2x2 Block für die Zielmode)
    V_MM = state.covariance[idx_m : idx_m + 2, idx_m : idx_m + 2]
    V_MR = state.covariance[idx_m : idx_m + 2, remaining_indices]
    V_RM = V_MR.T
    V_RR = state.covariance[np.ix_(remaining_indices, remaining_indices)]
    
    # Injektion des minimalen Heisenberg-Vakuumrauschens (0.5 * I_2)
    V_eff = V_MM + 0.5 * np.eye(2)
    V_eff_inv = np.linalg.inv(V_eff)

    # Multivariates Sampling über die verrauschte Verteilung
    if outcome is None:
        rng = rng if rng is not None else np.random.default_rng()
        measured_vector = rng.multivariate_normal(mean=state.displacement[idx_m : idx_m + 2], cov=V_eff)
    else:
        measured_vector = np.asarray(outcome, dtype=float)

    # Matrix-Konditionierung über das verrauschte Schur-Komplement
    d_cond = state.displacement[remaining_indices] + V_RM @ V_eff_inv @ (measured_vector - state.displacement[idx_m : idx_m + 2])
    V_cond = V_RR - V_RM @ V_eff_inv @ V_MR

    remaining_modes = tuple(m for m in state.modes if m != measured_mode)
    return measured_vector, GaussianState(remaining_modes, d_cond, V_cond)
```

=== Warum der Code das tut (Physikalische Kausalität)
- *`V_eff = V_MM + 0.5 * np.eye(2)`*: Das hinzugefügte `0.5 * np.eye(2)` repräsentiert exakt das Fluktuations-Quant des ungenutzten Strahlteiler-Eingangs. Ohne diesen Term wäre die resultierende Matrix-Inversion für ideal gequetschte Zustände singulär oder physikalisch unterspezifiziert, was zu Verletzungen der Robertson-Schrödinger-Unschärferelation im verbleibenden Zustand führen würde.
- *`V_cond = V_RR - V_RM @ V_eff_inv @ V_MR`*: Da durch das Messrauschen weniger Information über das System gewonnen werden kann als bei einer homodynen Messung, sorgt das modifizierte `V_eff_inv` dafür, dass die Varianzen des Restsystems $V_("cond")$ weniger stark schrumpfen. Die Eigenwerte der resultierenden Kovarianzmatrix bleiben garantierbar oberhalb des Vakuumlimits ($>= 0.5$).

---

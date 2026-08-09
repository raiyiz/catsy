from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy.linalg


@dataclass
class GaussianState:
    modes: tuple[str, ...]
    displacement: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        n_modes = len(self.modes)
        expected_dim = 2 * n_modes
        assert self.displacement.shape == (
            expected_dim,
        ), f"Displacement muss Länge {expected_dim} haben."
        assert self.covariance.shape == (
            expected_dim,
            expected_dim,
        ), f"Covariance muss {expected_dim}x{expected_dim} sein."

    def get_mode_index(self, mode_name: str) -> int:
        if mode_name not in self.modes:
            raise ValueError(
                f"Mode '{mode_name}' ist in diesem Zustand nicht vorhanden."
            )
        return self.modes.index(mode_name) * 2

    def to_qutip(self, N_cutoff: int = 15) -> qt.Qobj:
        """
        Konvertiert die Kovarianzmatrix V und den d-Vektor exakt in ein QuTiP
        Qobj (Dichtematrix rho). Nutzt das Williamson-Theorem zur Zersetzung in
        thermische Zustände + unitäre Transformationen.
        """
        n_modes = len(self.modes)

        # 1. Konstruiere die fundamentale symplektische Matrix Omega (J)
        # Omega = \bigoplus_{i=1}^n [[0, 1], [-1, 0]]
        omega_1 = np.array([[0, 1], [-1, 0]])
        Omega = scipy.linalg.block_diag(*[omega_1 for _ in range(n_modes)])

        # 2. Berechne die symplektischen Eigenwerte über die Matrix: i * Omega * V
        # Die Eigenwerte dieses Operators kommen paarweise als (+- nu_k) vor
        M = 1j * Omega @ self.covariance
        eigvals = np.linalg.eigvals(M)

        # Nur die positiven Imaginärteile extrahieren und sortieren
        nu = sorted(np.abs(eigvals.real)[::2])

        # 3. Thermische Zustände (Basis-Zustand rho_0) im Hilbertraum aufbauen
        # Ein reiner Vakuumzustand hat nu_k = 0.5. Größere Werte bedeuten
        # thermisches Rauschen.
        rho_list = []
        for nu_k in nu:
            if nu_k < 0.499:  # Numerischer Sanity-Check
                nu_k = 0.5
            n_thermal = nu_k - 0.5
            if n_thermal < 1e-6:
                # Reiner Vakuumzustand als Dichtematrix
                rho_list.append(qt.ket2dm(qt.fock(N_cutoff, 0)))
            else:
                # Thermischer Zustand
                rho_list.append(qt.thermal_dm(N_cutoff, n_thermal))

        rho_0 = qt.tensor(*rho_list)  # Gesamter unkorrelierter Ausgangszustand

        # 4. Standard-Kanonische Operatoren (a, x, p) im kombinierten Hilbertraum bauen
        a_ops = []
        for i in range(n_modes):
            op_list = [qt.qeye(N_cutoff)] * n_modes
            op_list[i] = qt.destroy(N_cutoff)
            a_ops.append(qt.tensor(*op_list))

        r_ops = []
        for a in a_ops:
            x = (a + a.dag()) / np.sqrt(2)
            p = (a - a.dag()) / (1j * np.sqrt(2))
            r_ops.extend([x, p])

        # 5. Berechnung der symplektischen Transformation S im Phasenraum
        # Wir finden die Matrix S, so dass S * V_diag * S^T = V.
        # Da QuTiP Operatoren exponentiell generiert, nutzen wir den
        # quadratischen Generator G: S = exp(Omega * G). G bestimmt den
        # Hamiltonoperator für das Squeezing/Mischen.
        V_diag = scipy.linalg.block_diag(*[nu_k * np.eye(2) for nu_k in nu])

        # Numerische Berechnung des Generators der Transformation über
        # Matrix-Logarithmus. Da V positiv definit ist, können wir über
        # Matrixwurzeln arbeiten
        V_diag_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(V_diag))
        X_mat = V_diag_inv_sqrt @ self.covariance @ V_diag_inv_sqrt

        # Symplektische Matrix S bestimmen
        S = scipy.linalg.sqrtm(self.covariance @ np.linalg.inv(V_diag)).real

        # Generator H_quad extrahieren: S = exp(Omega * G) -> G = -Omega * logm(S)
        G = -Omega @ scipy.linalg.logm(S).real

        # Baue den zugehörigen quantenmechanischen
        # Hamiltonoperator aus G_ij * r_i * r_j
        H_cv = 0
        for i in range(2 * n_modes):
            for j in range(2 * n_modes):
                if np.abs(G[i, j]) > 1e-9:
                    H_cv += 0.5 * G[i, j] * r_ops[i] * r_ops[j]

        # Wende die unitäre Transformation (Squeezing/Beamsplitting) auf rho_0 an
        U_cv = (-1j * H_cv).expm()
        rho_transformed = U_cv * rho_0 * U_cv.dag()

        # 6. Verschiebung (Displacement d) anwenden
        # D(alpha) = exp(alpha * a^dagger - alpha^* * a)
        # In Quadraturen ausgedrückt: exp(-i * (d_p * x - d_x * p))
        H_disp = 0
        for i in range(n_modes):
            d_x = self.displacement[2 * i]
            d_p = self.displacement[2 * i + 1]
            if np.abs(d_x) > 1e-9 or np.abs(d_p) > 1e-9:
                # Korrekte konventionelle Verschiebung über die x und p Operatoren
                H_disp += d_p * r_ops[2 * i] - d_x * r_ops[2 * i + 1]

        if H_disp != 0:
            D_cv = (-1j * H_disp).expm()
            rho_final = D_cv * rho_transformed * D_cv.dag()
        else:
            rho_final = rho_transformed

        return rho_final

    def plot_covariance(self):
        """Visualisiert die Korrelationen zwischen allen registrierten Moden."""
        plt.figure(figsize=(6, 5))
        ticks = []
        for m in self.modes:
            ticks.extend([f"q_{m}", f"p_{m}"])

        im = plt.imshow(self.covariance, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Varianz / Kovarianz")
        plt.xticks(range(len(ticks)), ticks)
        plt.yticks(range(len(ticks)), ticks)
        plt.title("QBS Multi-Mode Kovarianzmatrix V")


class GaussianOperations:
    @staticmethod
    def create_vacuum(modes: tuple[str, ...]) -> GaussianState:
        """Erzeugt einen Multi-Moden-Vakuumzustand (V = 0.5 * I)."""
        dim = 2 * len(modes)
        d = np.zeros(dim)
        V = 0.5 * np.eye(dim)  # Quantenrauschen-Limit nach deiner Konvention
        return GaussianState(modes=modes, displacement=d, covariance=V)

    @staticmethod
    def apply_squeezing(
        state: GaussianState, mode: str, r: float, theta: float = 0.0
    ) -> GaussianState:
        """Wendet Single-Mode Squeezing auf eine bestimmte Mode an."""
        idx = state.get_mode_index(mode)
        dim = len(state.displacement)

        # Lokale Squeezing-Matrix bauen
        S_local = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        # Rotation für die Phase theta
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        S_local = R @ S_local @ R.T

        # Symplektische Gesamt-Transformationsmatrix S bauen
        S_global = np.eye(dim)
        S_global[idx : idx + 2, idx : idx + 2] = S_local

        # Zustand updaten nach d' = S*d und V' = S*V*S^T
        new_d = S_global @ state.displacement
        new_V = S_global @ state.covariance @ S_global.T
        return GaussianState(
            modes=state.modes,
            displacement=new_d,
            covariance=new_V,
        )

    @staticmethod
    def apply_beam_splitter(
        state: GaussianState, mode_a: str, mode_b: str, eta: float
    ) -> GaussianState:
        """
        Transformiert zwei Moden an einem verlustfreien Beamsplitter.
        Nutzt deine symplektische Formulierung aus dem Dokument.
        """
        idx_a = state.get_mode_index(mode_a)
        idx_b = state.get_mode_index(mode_b)
        dim = len(state.displacement)

        t = np.sqrt(eta)
        r_coeff = np.sqrt(1 - eta)

        # Globale Transformationsmatrix S_BS aufbauen
        S_BS = np.eye(dim)

        # S_BS Blockstruktur für die Kopplung injizieren
        I2 = np.eye(2)
        S_BS[idx_a : idx_a + 2, idx_a : idx_a + 2] = t * I2
        S_BS[idx_a : idx_a + 2, idx_b : idx_b + 2] = r_coeff * I2
        S_BS[idx_b : idx_b + 2, idx_a : idx_a + 2] = -r_coeff * I2
        S_BS[idx_b : idx_b + 2, idx_b : idx_b + 2] = t * I2

        # Transformation ausführen
        new_d = S_BS @ state.displacement
        new_V = S_BS @ state.covariance @ S_BS.T
        return GaussianState(
            modes=state.modes,
            displacement=new_d,
            covariance=new_V,
        )

    @staticmethod
    def apply_loss(state: GaussianState, mode: str, eta: float) -> GaussianState:
        """Fügt einer Mode Dämpfung (Vakuum-Kopplung) hinzu."""
        idx = state.get_mode_index(mode)
        dim = len(state.displacement)

        # Dämpfungsmatrizen X und Y initialisieren
        X = np.eye(dim)
        X[idx : idx + 2, idx : idx + 2] = np.sqrt(eta) * np.eye(2)

        Y = np.zeros((dim, dim))
        Y[idx : idx + 2, idx : idx + 2] = (
            (1 - eta) * 0.5 * np.eye(2)
        )  # Vakuum-Rauschen kommt rein

        # Gesetz: d' = X*d und V' = X*V*X^T + Y
        new_d = X @ state.displacement
        new_V = X @ state.covariance @ X.T + Y
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)


@dataclass
class GaussianChannel:
    """
    Repräsentiert einen allgemeinen Gaußschen Kanal (General Gaussian Channel),
    der auf ein Subsystem aus bestimmten Moden wirkt.
    """

    target_modes: tuple[str, ...]  # Moden, auf die der Kanal wirkt
    X: np.ndarray  # Deterministische Transformation (Dämpfung/Verstärkung)
    Y: np.ndarray  # Symmetrische Rauschmatrix (Thermisches/Quantenrauschen)
    d0: np.ndarray  # Verschiebungs-Vektor (z.B. für kohärentes Treiben/Verschieben)

    def __post_init__(self):
        dim = 2 * len(self.target_modes)
        assert self.X.shape == (dim, dim), (
            f"X-Matrix muss Dimension {dim}x{dim} haben.",
        )
        assert self.Y.shape == (dim, dim), (
            f"Y-Matrix muss Dimension {dim}x{dim} haben.",
        )
        assert self.d0.shape == (dim,), f"d0-Vektor muss Länge {dim} haben."

    def apply(self, state: GaussianState) -> GaussianState:
        """Wendet den Kanal auf den globalen Zustand an."""
        global_dim = len(state.displacement)

        # Erstelle globale Transformationsmatrizen (Einheitsmatrizen als Basis)
        X_global = np.eye(global_dim)
        Y_global = np.zeros((global_dim, global_dim))
        d0_global = np.zeros(global_dim)

        # Injiziere die lokalen Matrizen an den richtigen Moden-Indizes
        for local_idx_mode1, m1 in enumerate(self.target_modes):
            global_idx_mode1 = state.get_mode_index(m1)

            # d0 injizieren
            d0_global[global_idx_mode1 : global_idx_mode1 + 2] = self.d0[
                local_idx_mode1 * 2 : local_idx_mode1 * 2 + 2
            ]

            for local_idx_mode2, m2 in enumerate(self.target_modes):
                global_idx_mode2 = state.get_mode_index(m2)

                # X injizieren
                X_global[
                    global_idx_mode1 : global_idx_mode1 + 2,
                    global_idx_mode2 : global_idx_mode2 + 2,
                ] = self.X[
                    local_idx_mode1 * 2 : local_idx_mode1 * 2 + 2,
                    local_idx_mode2 * 2 : local_idx_mode2 * 2 + 2,
                ]

                # Y injizieren
                Y_global[
                    global_idx_mode1 : global_idx_mode1 + 2,
                    global_idx_mode2 : global_idx_mode2 + 2,
                ] = self.Y[
                    local_idx_mode1 * 2 : local_idx_mode1 * 2 + 2,
                    local_idx_mode2 * 2 : local_idx_mode2 * 2 + 2,
                ]

        # Die globalen Transformationsgesetze anwenden
        new_d = X_global @ state.displacement + d0_global
        new_V = X_global @ state.covariance @ X_global.T + Y_global

        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)


class QBSChannels:
    """Fabrik-Klasse für standardmäßige optische Rauschkanäle."""

    @staticmethod
    def thermal_loss(mode: str, eta: float, n_thermal: float) -> GaussianChannel:
        """
        Kombiniert Dämpfung (Transmissivität eta) mit thermischem Rauschen (n_thermal).
        Perfekt für Faserdämpfung bei Raumtemperatur oder unvollständige Detektoren.
        """
        X = np.sqrt(eta) * np.eye(2)
        # Rauschen der Umgebung: V_env = (n_thermal + 0.5) * I
        V_env = (n_thermal + 0.5) * np.eye(2)
        # Y = (1 - eta) * V_env
        Y = (1 - eta) * V_env
        d0 = np.zeros(2)
        return GaussianChannel(target_modes=(mode,), X=X, Y=Y, d0=d0)

    @staticmethod
    def classical_phase_jitter(mode: str, sigma_phi: float) -> GaussianChannel:
        """
        Simuliert Phasenfluktuationen (Jitter) auf einer Faser/einem Spiegel.
        Führt im Zeitmittel zu einer Dekohärenz (Vergrößerung der p-Varianz im
                                                   rotierenden Bezugssystem).
        """
        # Für kleine Fluktuationen approximieren wir das als zusätzliches
        # Rauschen in der Phase
        X = np.eye(2)
        # Erzeugt zusätzliches Phasenrauschen proportional zur Varianz des
        # Jitters
        Y = np.array([[0, sigma_phi**2]])
        d0 = np.zeros(2)
        return GaussianChannel(target_modes=(mode,), X=X, Y=Y, d0=d0)

    @staticmethod
    def correlated_thermal_noise(
        mode_a: str, mode_b: str, eta: float, n_thermal: float, c_correlation: float
    ) -> GaussianChannel:
        """
        Erzeugt korreliertes thermisches Rauschen auf zwei Moden parallel.
        Nützlich, wenn zwei Kanäle thermisch an dieselbe Umgebung koppeln
        (z.B. im selben Faserstrang).
        """
        X = np.sqrt(eta) * np.eye(4)

        # Blockstruktur für Y aufbauen
        V_diag = (1 - eta) * (n_thermal + 0.5) * np.eye(2)
        V_cross = (1 - eta) * c_correlation * np.eye(2)

        Y = np.block([[V_diag, V_cross], [V_cross.T, V_diag]])
        d0 = np.zeros(4)
        return GaussianChannel(target_modes=(mode_a, mode_b), X=X, Y=Y, d0=d0)


from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CircuitOperation:
    """Repräsentiert ein abstraktes optisches Element oder einen Kanal im Circuit."""

    name: str
    modes: tuple[str, ...]
    kwargs: dict[str, Any]
    # Die tatsächliche Funktion aus GaussianOperations oder QBSChannels
    func: Callable[..., GaussianState] | Callable[..., GaussianChannel]


@dataclass
class GaussianCircuit:
    """
    Kompiliert und sequenziert eine Kette von Gaußschen Operationen (Gatter & Kanäle)
    und wendet sie auf ein registriertes Moden-Set an.
    """

    modes: tuple[str, ...] = field(default_factory=tuple)
    _operations: list[CircuitOperation] = field(default_factory=list, init=False)

    def add_mode(self, mode_name: str):
        """Fügt dem Circuit eine neue optische Mode hinzu."""
        if mode_name in self.modes:
            raise ValueError(f"Mode '{mode_name}' ist bereits im Circuit registriert.")
        self.modes = self.modes + (mode_name,)
        return self

    def squeeze(self, mode: str, r: float, theta: float = 0.0):
        """Fügt ein Single-Mode Squeezing-Gatter hinzu."""
        self._operations.append(
            CircuitOperation(
                name="Squeezing",
                modes=(mode,),
                kwargs={"r": r, "theta": theta},
                func=GaussianOperations.apply_squeezing,
            )
        )
        return self  # Erlaubt Method Chaining (.squeeze().beam_splitter())

    def beam_splitter(self, mode_a: str, mode_b: str, eta: float):
        """Fügt einen Beamsplitter hinzu."""
        self._operations.append(
            CircuitOperation(
                name="BeamSplitter",
                modes=(mode_a, mode_b),
                kwargs={"eta": eta},
                func=GaussianOperations.apply_beam_splitter,
            )
        )
        return self

    def loss(self, mode: str, eta: float):
        """Fügt reinen Vakuum-Verlust hinzu."""
        self._operations.append(
            CircuitOperation(
                name="Loss",
                modes=(mode,),
                kwargs={"eta": eta},
                func=GaussianOperations.apply_loss,
            )
        )
        return self

    def thermal_loss(self, mode: str, eta: float, n_thermal: float):
        """Fügt einen allgemeinen thermischen Rauschkanal hinzu."""
        # Da dies ein Kanal ist, nutzen wir die Fabrik aus QBSChannels
        self._operations.append(
            CircuitOperation(
                name="ThermalLossChannel",
                modes=(mode,),
                kwargs={"eta": eta, "n_thermal": n_thermal},
                func=QBSChannels.thermal_loss,
            )
        )
        return self

    def compile_and_run(self, initial_state: GaussianState = None) -> GaussianState:
        """
        Validiert alle Operationen gegen die registrierten Moden
        und führt die gesamte Kette sequenziell aus.
        """
        if not self.modes:
            raise ValueError("Der Circuit enthält keine registrierten Moden.")

        # Wenn kein Anfangszustand übergeben wurde, starten wir im globalen Vakuum
        if initial_state is None:
            current_state = GaussianOperations.create_vacuum(self.modes)
        else:
            # Sicherstellen, dass die Moden des Zustands mit dem Circuit übereinstimmen
            assert set(initial_state.modes) == set(
                self.modes
            ), "Moden des Initialzustands passen nicht zum Circuit."
            current_state = initial_state

        print(f"🚀 Starte QBS-Circuit Execution für Moden: {self.modes}")
        print(f"Anzahl geplanter Operationen: {len(self._operations)}\n")

        for idx, op in enumerate(self._operations):
            # Validierung: Prüfen, ob alle Zielmoden im Circuit existieren
            for m in op.modes:
                if m not in self.modes:
                    raise ValueError(
                        f"Fehler bei Op #{idx} ({op.name}): Mode '{m}' ist "
                        "nicht im Circuit registriert!"
                    )

            # Ausführen je nach Typ der Funktion
            if op.name.endswith("Channel"):
                # Wenn es ein eigenständiger GaussianChannel ist, instanziieren
                # wir ihn und wenden ihn an
                channel_instance = op.func(mode=op.modes[0], **op.kwargs)
                current_state = channel_instance.apply(current_state)
                print(
                    f" [{idx+1}/{len(self._operations)}] Angewendet: "
                    f"{op.name} auf {op.modes}"
                )
            else:
                # Standard-Gatter (In-Place Transformation)
                if len(op.modes) == 1:
                    current_state = op.func(
                        current_state, mode=op.modes[0], **op.kwargs
                    )
                elif len(op.modes) == 2:
                    current_state = op.func(
                        current_state,
                        mode_a=op.modes[0],
                        mode_b=op.modes[1],
                        **op.kwargs,
                    )
                print(
                    f" [{idx+1}/{len(self._operations)}] Ausgeführt: "
                    f"{op.name} auf {op.modes}"
                )

        print("\n Execution erfolgreich beendet.")
        return current_state


class GaussianMeasurements:
    @staticmethod
    def homodyne_measurement(
        state: GaussianState, measured_mode: str, phi: float, outcome: float = None
    ) -> tuple[float, GaussianState]:
        """
        Führt eine Homodyn-Messung auf einer bestimmten Mode durch.

        Parameters:
        -----------
        state : GaussianState
            Der aktuelle globale Multi-Moden-Zustand.
        measured_mode : str
            Die Mode, die gemessen wird (z.B. 'a').
        phi : float
            Die Phase des Lokaloszillators (0 = misst x/q, pi/2 = misst p).
        outcome : float, optional
            Falls vorgegeben, wird dieses Messergebnis erzwungen.
            Falls None, wird das Ergebnis statistisch korrekt aus der
            Wahrscheinlichkeitsverteilung gewürfelt.

        Returns:
        --------
        measured_value : float
            Das (gewürfelte oder übergebene) reelle Messergebnis.
        conditional_state : GaussianState
            Der kollabierte Zustand der verbleibenden Moden nach der Messung.
        """
        n_modes = len(state.modes)
        idx_m = state.get_mode_index(measured_mode)

        # 1. Rotiere den Zustand temporär, damit die gemessene Quadratur auf 'x' liegt
        # Das vereinfacht die Mathematik (wir messen immer 'x' nach der Rotation)
        R_local = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        R_global = np.eye(2 * n_modes)
        R_global[idx_m : idx_m + 2, idx_m : idx_m + 2] = R_local

        d_rot = R_global @ state.displacement
        V_rot = R_global @ state.covariance @ R_global.T

        # 2. Partitioniere d_rot und V_rot in gemessene Mode (M) und
        # verbleibende Moden (R)
        # Indizes für die gemessene x-Quadratur
        idx_x = idx_m
        # Alle anderen Indizes (verbleibende Quadraturen)
        remaining_indices = [
            i for i in range(2 * n_modes) if i != idx_x and i != idx_m + 1
        ]

        # Kovarianz-Blöcke extrahieren
        V_MM = V_rot[idx_x, idx_x]  # Varianz der gemessenen Quadratur (Skalar)
        V_MR = V_rot[idx_x, remaining_indices]  # Spaltenvektor (Beziehung M zu R)
        V_RM = V_rot[remaining_indices, idx_x]  # Zeilenvektor
        V_RR = V_rot[
            np.ix_(remaining_indices, remaining_indices)
        ]  # Verbleibende Kovarianz

        # Displacements extrahieren
        d_M = d_rot[idx_x]
        d_R = d_rot[remaining_indices]

        # 3. Messergebnis bestimmen (falls nicht vorgegeben, aus Gaußverteilung würfeln)
        if outcome is None:
            # Standardabweichung ist die Wurzel aus der Varianz V_MM
            sigma = np.sqrt(V_MM)
            measured_value = np.random.normal(loc=d_M, scale=sigma)
        else:
            measured_value = outcome

        # 4. Kollaps-Gleichungen (Schur-Komplement) anwenden
        # Der verbleibende d-Vektor verschiebt sich basierend auf dem Messergebnis!
        # d_cond = d_R + V_RM * (1/V_MM) * (measured_value - d_M)
        d_cond_rot = d_R + V_RM * (1.0 / V_MM) * (measured_value - d_M)

        # Die neue Kovarianzmatrix schrumpft (Squeezing durch Messung!)
        # V_cond = V_RR - V_RM * (1/V_MM) * V_MR
        V_cond_rot = V_RR - np.outer(V_RM, V_MR) / V_MM

        # 5. Zurück-Rotation der verbleibenden Moden ist nicht nötig, da wir
        # sie nicht angefasst haben.
        # Wir müssen nur die gemessene Mode aus der Liste der aktiven Moden löschen.
        remaining_modes = tuple(m for m in state.modes if m != measured_mode)

        return measured_value, GaussianState(
            modes=remaining_modes, displacement=d_cond_rot, covariance=V_cond_rot
        )


def plot_wigner_analytically(
    state: GaussianState, mode_name: str, x_max: float = 4.0, num_points: int = 150
):
    """
    Berechnet und plottet die Wigner-Funktion einer einzelnen Mode
    rein analytisch aus d und V, komplett ohne QuTiP-Hilbertraum.
    """
    # 1. Extrahiere das 2x2 Subsystem für die gewünschte Mode
    idx = state.get_mode_index(mode_name)

    d_mode = state.displacement[idx : idx + 2]  # [d_x, d_p]
    V_mode = state.covariance[idx : idx + 2, idx : idx + 2]  # 2x2 Kovarianzmatrix

    # 2. Erzeuge das 2D-Gitter für den Phasenraum
    xvec = np.linspace(-x_max, x_max, num_points)
    pvec = np.linspace(-x_max, x_max, num_points)
    X, P = np.meshgrid(xvec, pvec)

    # 3. Berechne die Wigner-Funktion punktweise (vektorisiert für Performance)
    det_V = np.linalg.det(V_mode)
    inv_V = np.linalg.inv(V_mode)

    # Verschiebung berechnen (Zentrieren um den d-Vektor)
    dX = X - d_mode[0]
    dP = P - d_mode[1]

    # Den Exponenten der Gauß-Verteilung bestimmen: r^T * V^-1 * r
    # Da inv_V eine 2x2 Matrix [[G00, G01], [G10, G11]] ist:
    exponent = (
        dX * inv_V[0, 0] * dX
        + dX * inv_V[0, 1] * dP
        + dP * inv_V[1, 0] * dX
        + dP * inv_V[1, 1] * dP
    )

    # Gaußsche Wigner-Funktion zusammensetzen
    W = (1.0 / (2.0 * np.pi * np.sqrt(det_V))) * np.exp(-0.5 * exponent)

    # 4. Plotten
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, P, W, 100, cmap="RdBu_r")
    plt.colorbar(contour, label="Wigner-Wahrscheinlichkeitsdichte")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.title(f"Analytische Wigner-Funktion für Mode '{mode_name}'")
    plt.xlabel("x (Ort / In-Phase Quadratur)")
    plt.ylabel("p (Impuls / Quadraturphase)")
    plt.axis("equal")
    plt.show()


def plot_joint_correlation(
    state: GaussianState, mode_a: str, mode_b: str, x_max: float = 3.0
):
    """Plottet die Wahrscheinlichkeitsverteilung von x_a vs x_b (EPR-Korrelation) im CV-Raum."""
    idx_a = state.get_mode_index(mode_a)
    idx_b = state.get_mode_index(mode_b)

    # Extrahiere die relevanten Sub-Kovarianzen für x_a und x_b
    # r = [x_a, p_a, x_b, p_b] -> x_a ist Index 0, x_b ist Index 2 im lokalen Verbund
    V_sub = np.array(
        [
            [state.covariance[idx_a, idx_a], state.covariance[idx_a, idx_b]],
            [state.covariance[idx_b, idx_a], state.covariance[idx_b, idx_b]],
        ]
    )
    d_sub = np.array([state.displacement[idx_a], state.displacement[idx_b]])

    # Gitter bauen
    xvec = np.linspace(-x_max, x_max, 150)
    X_a, X_b = np.meshgrid(xvec, xvec)

    # 2D-Gauß-Verteilung berechnen
    det_V = np.linalg.det(V_sub)
    inv_V = np.linalg.inv(V_sub)

    dX_a = X_a - d_sub[0]
    dX_b = X_b - d_sub[1]

    exponent = (
        inv_V[0, 0] * dX_a**2
        + (inv_V[0, 1] + inv_V[1, 0]) * dX_a * dX_b
        + inv_V[1, 1] * dX_b**2
    )
    P = (1.0 / (2.0 * np.pi * np.sqrt(det_V))) * np.exp(-0.5 * exponent)

    plt.figure(figsize=(6, 5))
    plt.contourf(X_a, X_b, P, 100, cmap="viridis")
    plt.colorbar(label="Wahrscheinlichkeitsdichte")
    plt.title(f"EPR-Korrelation: Quadratur $x_{mode_a}$ vs $x_{mode_b}$")
    plt.xlabel(f"x_{mode_a}")
    plt.ylabel(f"x_{mode_b}")
    plt.axis("equal")
    plt.show()

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
        assert self.displacement.shape == (expected_dim,), f"Displacement muss Länge {expected_dim} haben."
        assert self.covariance.shape == (expected_dim, expected_dim), f"Covariance muss {expected_dim}x{expected_dim} sein."

    def get_mode_index(self, mode_name: str) -> int:
        if mode_name not in self.modes:
            raise ValueError(f"Mode '{mode_name}' ist in diesem Zustand nicht vorhanden.")
        return self.modes.index(mode_name) * 2

    def to_qutip(self, N_cutoff: int = 15) -> qt.Qobj:
        """
        Konvertiert die Kovarianzmatrix V und den d-Vektor exakt in ein QuTiP Qobj (Dichtematrix rho).
        Nutzt das Williamson-Theorem zur Zersetzung in thermische Zustände + unitäre Transformationen.
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
        # Ein reiner Vakuumzustand hat nu_k = 0.5. Größere Werte bedeuten thermisches Rauschen.
        rho_list = []
        for nu_k in nu:
            if nu_k < 0.499: # Numerischer Sanity-Check
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
        # Da QuTiP Operatoren exponentiell generiert, nutzen wir den quadratischen Generator G:
        # S = exp(Omega * G). G bestimmt den Hamiltonoperator für das Squeezing/Mischen.
        V_diag = scipy.linalg.block_diag(*[nu_k * np.eye(2) for nu_k in nu])
        
        # Numerische Berechnung des Generators der Transformation über Matrix-Logarithmus
        # Da V positiv definit ist, können wir über Matrixwurzeln arbeiten
        V_diag_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(V_diag))
        X_mat = V_diag_inv_sqrt @ self.covariance @ V_diag_inv_sqrt
        
        # Symplektische Matrix S bestimmen
        S = scipy.linalg.sqrtm(self.covariance @ np.linalg.inv(V_diag)).real
        
        # Generator H_quad extrahieren: S = exp(Omega * G) -> G = -Omega * logm(S)
        G = -Omega @ scipy.linalg.logm(S).real
        
        # Baue den zugehörigen quantenmechanischen Hamiltonoperator aus G_ij * r_i * r_j
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
                H_disp += (d_p * r_ops[2 * i] - d_x * r_ops[2 * i + 1])
                
        if H_disp != 0:
            D_cv = (-1j * H_disp).expm()
            rho_final = D_cv * rho_transformed * D_cv.dag()
        else:
            rho_final = rho_transformed
            
        return rho_final
# # 
# @dataclass
# class GaussianState:
#     modes: tuple[str, ...]  # Registrierte Moden-Namen, z.B. ("a", "b")
#     displacement: np.ndarray  # d-Vektor (Länge 2 * N_modes)
#     covariance: np.ndarray  # V-Matrix (Dimension 2N_modes x 2N_modes)
# 
#     def __post_init__(self):
#         # Validierung der Dimensionen (2 Quadraturen pro Mode: x und p)
#         n_modes = len(self.modes)
#         expected_dim = 2 * n_modes
#         assert self.displacement.shape == (
#             expected_dim,
#         ), f"Displacement muss Länge {expected_dim} haben."
#         assert self.covariance.shape == (
#             expected_dim,
#             expected_dim,
#         ), f"Covariance muss {expected_dim}x{expected_dim} sein."
# 
#     def get_mode_index(self, mode_name: str) -> int:
#         """Gibt den Start-Index (für x) einer Mode im globalen Vektor zurück."""
#         if mode_name not in self.modes:
#             raise ValueError(
#                 f"Mode '{mode_name}' ist in diesem Zustand nicht vorhanden."
#             )
#         return self.modes.index(mode_name) * 2
# 
#     def to_qutip(self, N_cutoff: int = 15) -> qt.Qobj:
#         """
#         Konvertiert den Gaußschen Zustand bei Bedarf in eine QuTiP-Dichtematrix (Fock-Basis).
#         Nutzt die Weyl-Operator / charakteristische Funktion Methode.
#         """
#         n_modes = len(self.modes)
#         # Erstelle Vakuum im kombinierten Hilbertraum
#         vac_list = [qt.fock(N_cutoff, 0) for _ in range(n_modes)]
#         rho = qt.ket2dm(qt.tensor(*vac_list))
# 
#         # Vernichter im Gesamtraum bauen
#         a_ops = []
#         for i in range(n_modes):
#             op_list = [qt.qeye(N_cutoff)] * n_modes
#             op_list[i] = qt.destroy(N_cutoff)
#             a_ops.append(qt.tensor(*op_list))
# 
#         # Konstruiere Quadratur-Operatoren basierend auf deiner Konvention:
#         # x = (a + a^dagger)/sqrt(2), p = (a - a^dagger)/(i*sqrt(2))
#         r_ops = []
#         for a in a_ops:
#             x = (a + a.dag()) / np.sqrt(2)
#             p = (a - a.dag()) / (1j * np.sqrt(2))
#             r_ops.extend([x, p])
# 
#         # Da wir eine allgemeine Covariance-Matrix in QuTiP konvertieren wollen,
#         # ist der sauberste Weg die thermische/Squeezing-Transformation im Phasenraum.
#         # Für Demonstrationszwecke bauen wir hier den Zustand über eine Verschiebung und Squeezing.
#         # (Hinweis: Für ein volles Produktionstool würde man hier über die charakteristische Funktion gehen)
#         raise NotImplementedError(
#             "Ausfaltung beliebiger korrelierter Matrizen in die Fock-Basis "
#             "erfordert charakteristische Integration. Nutze reduzierte QuTiP-Zustände für Standard-Plots."
#         )

    def plot_covariance(self, do_plot=True):
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
        if do_plot:
            plt.show()


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
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        S_local = R @ S_local @ R.T

        # Symplektische Gesamt-Transformationsmatrix S bauen
        S_global = np.eye(dim)
        S_global[idx : idx + 2, idx : idx + 2] = S_local

        # Zustand updaten nach d' = S*d und V' = S*V*S^T
        new_d = S_global @ state.displacement
        new_V = S_global @ state.covariance @ S_global.T
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)

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
        return GaussianState(modes=state.modes, displacement=new_d, covariance=new_V)

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
        assert self.X.shape == (dim, dim), f"X-Matrix muss Dimension {dim}x{dim} haben."
        assert self.Y.shape == (dim, dim), f"Y-Matrix muss Dimension {dim}x{dim} haben."
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
        Führt im Zeitmittel zu einer Dekohärenz (Vergrößerung der p-Varianz im rotierenden Bezugssystem).
        """
        # Für kleine Fluktuationen approximieren wir das als zusätzliches Rauschen in der Phase
        X = np.eye(2)
        # Erzeugt zusätzliches Phasenrauschen proportional zur Varianz des Jitters
        Y = np.array([[0, sigma_phi**2]])
        d0 = np.zeros(2)
        return GaussianChannel(target_modes=(mode,), X=X, Y=Y, d0=d0)

    @staticmethod
    def correlated_thermal_noise(
        mode_a: str, mode_b: str, eta: float, n_thermal: float, c_correlation: float
    ) -> GaussianChannel:
        """
        Erzeugt korreliertes thermisches Rauschen auf zwei Moden parallel.
        Nützlich, wenn zwei Kanäle thermisch an dieselbe Umgebung koppeln (z.B. im selben Faserstrang).
        """
        X = np.sqrt(eta) * np.eye(4)

        # Blockstruktur für Y aufbauen
        V_diag = (1 - eta) * (n_thermal + 0.5) * np.eye(2)
        V_cross = (1 - eta) * c_correlation * np.eye(2)

        Y = np.block([[V_diag, V_cross], [V_cross.T, V_diag]])
        d0 = np.zeros(4)
        return GaussianChannel(target_modes=(mode_a, mode_b), X=X, Y=Y, d0=d0)

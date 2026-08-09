import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt
from qutip.visualization import matrix_histogram

from .states import (
    GaussianCircuit,
    GaussianMeasurements,
    GaussianOperations,
    NonGaussianOperations,
    QBSChannels,
    plot_joint_correlation,
    plot_wigner_analytically,
)


def test_covariance_tmsv():
    # 1. Erstelle ein leeres System mit zwei Moden "a" und "b"
    state = GaussianOperations.create_vacuum(modes=("a", "b"))

    # 2. Squeezen der Moden in entgegengesetzte Richtungen (r = 0.6)
    # Mode A wird in q gesqueezed, Mode B in p
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.6, theta=0.0)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.6, theta=np.pi / 2)

    print("Kovarianz V vor dem Beam Splitter (Moden sind unabhängig):")
    print(np.round(state.covariance, 3))

    # 3. Beide Moden auf den 50:50 Beam Splitter schicken (eta = 0.5)
    state = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )

    print("\nKovarianz V nach dem Beam Splitter (Verschränkung ist entstanden!):")
    print(np.round(state.covariance, 3))

    # 4. Plotten der resultierenden Kovarianzmatrix
    # state.plot_covariance()
    print("SKIPPING PLOT")


def test_cv_chan_to_fock():

    # 1. Erzeuge einen verschränkten Zustand mit Rauschen im CV-Formalismus
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.5)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.5, theta=np.pi / 2)
    state = GaussianOperations.apply_beam_splitter(
        state, mode_a="a", mode_b="b", eta=0.5
    )

    # Füge etwas thermischen Verlust hinzu (Kanal aus dem vorherigen Schritt)
    loss = QBSChannels.thermal_loss(mode="a", eta=0.9, n_thermal=0.2)
    noisy_cv_state = loss.apply(state)

    # 2. KONVERTIERUNG IN QUTIP
    # Wir nutzen N_cutoff = 12 (reicht für r=0.5 dicke aus)
    rho_qutip = noisy_cv_state.to_qutip(N_cutoff=12)

    print("--- QuTiP Konvertierung erfolgreich ---")
    print(f"Typ des Objekts: {type(rho_qutip)}")
    print(f"Dimensionen im Hilbertraum: {rho_qutip.dims}")

    # 3. Physikalischen Test machen: Berechne Verschränkungs-Entropie in QuTiP
    rho_a = rho_qutip.ptrace(0)
    entropy = qt.entropy_vn(rho_a)
    print(f"Von-Neumann-Entropie der Mode A (berechnet via QuTiP): {entropy:.4f}")

    # 4. Plot der Wigner-Funktion direkt aus dem konvertierten QuTiP Qobj
    xvec = np.linspace(-3, 3, 100)
    W = qt.wigner(rho_a, xvec, xvec)

    plt.figure(figsize=(5, 4))
    plt.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
    plt.colorbar()
    plt.title("Wigner-Funktion (Mode A) aus konvertiertem QuTiP Objekt")
    # plt.show()


def test_qo_epr():

    # 1. Circuit definieren und Moden anmelden
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.add_mode("b")

    # 2. Pipeline deklarativ aufbauen (Method Chaining)
    circuit.squeeze(mode="a", r=0.6, theta=0.0).squeeze(
        mode="b", r=0.6, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5).thermal_loss(
        mode="b", eta=0.7, n_thermal=0.3
    )  # Verlust auf dem Transportweg von Mode b

    # 3. Compiler triggern und Zustand berechnen
    final_cv_state = circuit.compile_and_run()

    # 4. Kovarianz-Ergebnis im Continuous-Variable Raum plotten
    print("SKIPPING PLOT")
    # final_cv_state.plot_covariance()

    # 5. Voller quantenmechanischer Test: Konvertierung in QuTiP Hilbertraum
    # dank deines Williamson-Theorems!
    print("\n--- Analysiere Endzustand in QuTiP ---")
    rho_qutip = final_cv_state.to_qutip(N_cutoff=15)

    # Berechne Reinheit (Purity) des Gesamtzustands tr(rho^2)
    # Durch den thermischen Kanal sollte das System nicht mehr rein (=1) sein
    purity = (rho_qutip * rho_qutip).tr()
    print(
        f"Reinheit des Gesamtsystems nach Verlusten: {purity.real:.4f} (< 1.0 "
        "bedeutet gemischter Zustand)"
    )

    # Zeige, dass Mode a und b immer noch Korrelationen besitzen
    entropy_a = qt.entropy_vn(rho_qutip.ptrace(0))
    print(f"Von-Neumann-Entropie Subsystem A: {entropy_a:.4f}")


def test_measure_homodyne():

    # 1. Erzeuge verschränkten EPR-Zustand über unseren Circuit
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=1.0).squeeze(
        mode="b", r=1.0, theta=np.pi / 2
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)

    epr_state = circuit.compile_and_run()

    print(f"Ursprünglicher d-Vektor: {epr_state.displacement} (Beide im Ursprung)")

    # 2. Wir simulieren eine Homodyn-Messung auf Mode 'a' (phi=0 bedeutet x-Messung)
    # Wir erzwingen ein extremes Messergebnis von x = +2.5 (Standard wäre um die 0)
    val, collapsed_state = GaussianMeasurements.homodyne_measurement(
        epr_state, measured_mode="a", phi=0.0, outcome=2.5
    )

    print("\n--- MESSUNG AUSGEFÜHRT ---")
    print(f"Gemessener Wert auf Mode 'a': {val}")
    print(f"Verbleibende Moden im System: {collapsed_state.modes}")
    print(f"Neuer d-Vektor von Mode 'b': {np.round(collapsed_state.displacement, 3)}")

    # 3. Zum Vergleich: Was passiert, wenn wir x = -2.5 gemessen hätten?
    _, collapsed_state_neg = GaussianMeasurements.homodyne_measurement(
        epr_state, measured_mode="a", phi=0.0, outcome=-2.5
    )
    print(
        "Neuer d-Vektor von Mode 'b' bei negativem Messergebnis: "
        f"{np.round(collapsed_state_neg.displacement, 3)}"
    )


def test_wigner_analytical_plotting():

    # 1. Setup: Ein Circuit mit extremem Squeezing und einer Verschiebung
    circuit = GaussianCircuit()
    circuit.add_mode("a")

    # Wir squeezen massiv (r=1.8) und verschieben die Mode
    # im Phasenraum (x=2.0, p=1.0)
    circuit.squeeze(mode="a", r=1.8, theta=0.0)

    # Manuelle Verschiebung im d-Vektor injizieren für den Test
    test_state = circuit.compile_and_run()
    test_state.displacement[0] = 2.0  # d_x
    test_state.displacement[1] = 1.0  # d_p

    # 2. Plot aufrufen
    print("Generiere Wigner-Plot instantan...")
    print("SKIPPING PLOT")
    # plot_wigner_analytically(test_state, mode_name="a", x_max=5.0)


@pytest.mark.skip
def test_wigner_qutip_plotting():

    # 1. Setup: Verschränkten EPR-Zustand im CV-Circuit erzeugen
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6).squeeze(
        mode="b", r=0.6, theta=np.pi
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)

    cv_state = circuit.compile_and_run()

    # --- ANSATZ 1: Die CV-Korrelation direkt plotten ---
    plot_joint_correlation(cv_state, "a", "b")
    plt.show()

    # =====================================================================
    # --- ANSATZ 2: Konvertierung und native QuTiP-Plots nutzen ---
    # =====================================================================
    # Wir konvertieren den Zustand in die QuTiP Fock-Basis (Cutoff=12 reicht für r=0.6)
    rho_qt = cv_state.to_qutip(N_cutoff=12)

    # --- A) Native QuTiP Wigner-Funktion plotten ---
    # Wir werfen per ptrace(1) Mode b weg, um die Wigner-Funktion von Mode a zu sehen
    rho_a = rho_qt.ptrace(0)

    xvec = np.linspace(-4, 4, 200)
    # Das ist die native QuTiP Berechnungsfunktion [source: 1.1.2]
    W_qutip = qt.wigner(rho_a, xvec, xvec)

    # Plotten mit Standard-Matplotlib (wie im QuTiP-Handbuch empfohlen) [source: 1.2.4]
    fig, ax = plt.subplots(figsize=(5, 4))
    cont = ax.contourf(xvec, xvec, W_qutip, 100, cmap="RdBu_r")
    fig.colorbar(cont, ax=ax)
    ax.set_title("Native QuTiP Wigner-Funktion (Mode A)")
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    plt.show()

    # --- B) Natives QuTiP Matrix-Histogramm (Fock-Besetzung) ---
    # Zeigt die Amplituden der Dichtematrix in der Teilchenzahl-Basis an [source: 1.1.1, 1.1.6]
    # Da es sich um ein 2-Moden-System handelt, reduzieren wir es auf Mode A für bessere Lesbarkeit
    fig, ax = matrix_histogram(rho_a)
    ax.view_init(azim=-30, elev=40)
    plt.title("Natives QuTiP Matrix-Histogramm der Mode A")
    plt.show()


def test_remove_phot_from_sqv():

    # 1. Erstelle ein einfaches, Single-Mode Squeezed Vacuum (rein Gaußsch)
    circuit = GaussianCircuit()
    circuit.add_mode("a")
    circuit.squeeze(mode="a", r=0.55)  # Squeezing vorbereiten
    gaussian_squeezed = circuit.compile_and_run()

    # 2. Wende die Nicht-Gaußsche Operation an (Photonen-Abzug)
    # Wir nutzen N_cutoff=25, um genügend numerischen Platz im Hilbertraum zu haben
    print("\nSubtrahiere ein Photon...")
    rho_cat = NonGaussianOperations.photon_subtraction(
        gaussian_squeezed, mode_name="a", N_cutoff=25
    )

    # 3. Berechne die Wigner-Funktion des neuen Nicht-Gaußschen Zustands
    xvec = np.linspace(-4, 4, 250)
    W_cat = qt.wigner(
        rho_cat, xvec, xvec
    )  # Nutzt das native QuTiP-Modul [source: 1.1.2]

    # 4. Plotten
    plt.figure(figsize=(6, 5))
    # Wir nutzen ein symmetrisches vmin/vmax, damit man die Negativität perfekt sieht
    contour = plt.contourf(xvec, xvec, W_cat, 100, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    plt.colorbar(contour, label="Wigner-Dichte")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.title(
        "Schrödinger-Katze via Photonen-Abzug\n(Beachte den blauen, negativen Kern!)"
    )
    plt.xlabel("x")
    plt.ylabel("p")
    plt.axis("equal")
    plt.show()


def test_full_cavity():

    # =====================================================================
    # 1. PHASENRAUM-VORBEREITUNG (Gaußscher Circuit)
    # =====================================================================
    # Wir bauen einen Circuit für eine Kavitäts-Mode 'c' auf.
    circuit = GaussianCircuit()
    circuit.add_mode("c")

    # Wir injizieren instantan ein moderates Squeezing beim Einkoppeln (r = 0.5)
    circuit.squeeze(mode="c", r=0.5, theta=0.0)
    initial_cv_state = circuit.compile_and_run()

    # =====================================================================
    # 2. HYBRIDE SCHNITTSTELLE: Übergang in den Hilbertraum
    # =====================================================================
    N_cutoff = 20  # Dimension des Kavitäts-Hilbertraums
    # Exakte Konvertierung via Williamson-Theorem in die QuTiP-Dichtematrix rho_0
    rho_0 = initial_cv_state.to_qutip(N_cutoff=N_cutoff)

    # =====================================================================
    # 3. KAVITÄTEN-ZEITENTWICKLUNG (QuTiP Master Equation Solver)
    # =====================================================================
    # Physikalische Kavitäten-Parameter definieren
    omega_c = 2.0 * np.pi * 1.0  # Kavitätsfrequenz (z.B. 1 GHz im rotierenden System)
    kappa = (
        0.3  # Zerfallsrate der Kavität (Photonenverlust durch Spiegel) [source: 1.3.1]
    )

    # Operatoren im Hilbertraum definieren
    a = qt.destroy(N_cutoff)
    H_cav = omega_c * a.dag() * a  # Freier Kavitäts-Hamiltonoperator

    # Lindblad Kollaps-Operatoren für dissipative Verluste definieren [source: 1.1.2, 1.3.1]
    # sqrt(kappa) * a beschreibt das kontinuierliche Heraussickern von Photonen [source: 1.3.1]
    collapse_operators = [np.sqrt(kappa) * a]

    # Zeitgitter für die Simulation (0 bis 5 Zeit-Einheiten)
    tlist = np.linspace(0, 5, 200)

    print("\n🎬 Starte zeitabhängige Lindblad-Kavitätensimulation in QuTiP...")
    # Berechne die offene Quantendynamik des Systems [source: 1.1.2]
    simulation_result = qt.mesolve(H_cav, rho_0, tlist, c_ops=collapse_operators)

    # Zustand am Ende der Zeitentwicklung extrahieren
    rho_t_final = simulation_result.states[-1]
    photon_numbers = [
        qt.expect(a.dag() * a, state) for state in simulation_result.states
    ]

    # =====================================================================
    # 4. NICHT-GAUSSSCHE INTERAKTION AM KAVITÄTSAUSGANG
    # =====================================================================
    print(
        "⚡ Kavitäten-Endzustand erreicht. Wende nicht-Gaußschen Photonen-Abzug an..."
    )
    # Ein Photon fliegt aus der Kavität und triggert unseren Detektor (Heralded State)
    # Wir simulieren den re-normierten Zustand nach der bedingten Messung
    # rho_subtracted = a * rho_t_final * a.dag()
    rho_subtracted = a.dag() * rho_t_final * a
    rho_final_non_gaussian = rho_subtracted / rho_subtracted.tr()

    # =====================================================================
    # 5. MULTI-PANEL VISUALISIERUNG DER DYNAMIK
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: Zeitliche Entwicklung der Photonenbesetzung in der Kavität
    axes[0].plot(
        tlist,
        photon_numbers,
        label=r"$\langle n \rangle$ (Photonenzahl)",
        color="darkblue",
        lw=2,
    )
    axes[0].set_title("Photonenverlust der Kavität über die Zeit")
    axes[0].set_xlabel("Zeit $t$")
    axes[0].set_ylabel("Mittlere Photonenzahl")
    axes[0].grid(True, ls="--")
    axes[0].legend()

    # Plot B: Wigner-Funktion direkt VOR dem Photonen-Abzug (Gaußscher, gedämpfter Zustand)
    xvec = np.linspace(-4, 4, 200)
    W_before = qt.wigner(rho_t_final, xvec, xvec)
    cont1 = axes[1].contourf(
        xvec, xvec, W_before, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3
    )
    fig.colorbar(cont1, ax=axes[1])
    axes[1].set_title("Vor Photonen-Abzug\n(Klassisch-gedämpftes Ellipsoid)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p")
    axes[1].axis("equal")

    # Plot C: Wigner-Funktion NACH dem Photonen-Abzug (Nicht-Gaußsche Schrödinger-Katze)
    W_after = qt.wigner(rho_final_non_gaussian, xvec, xvec)
    cont2 = axes[2].contourf(
        xvec, xvec, W_after, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3
    )
    fig.colorbar(cont2, ax=axes[2])
    axes[2].set_title("Nach Photonen-Abzug\n(Nicht-Gaußsche Wigner-Negativität!)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("p")
    axes[2].axis("equal")

    plt.tight_layout()
    plt.show()

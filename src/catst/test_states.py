from time import perf_counter

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
    QBSSimulator,
    plot_joint_correlation,
    plot_wigner_analytically,
)

PLOT = False

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

    if PLOT:
        state.plot_covariance()


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
    if PLOT:
        xvec = np.linspace(-3, 3, 100)
        W = qt.wigner(rho_a, xvec, xvec)

        plt.figure(figsize=(5, 4))
        plt.contourf(xvec, xvec, W, 100, cmap="RdBu_r")
        plt.colorbar()
        plt.title("Wigner-Funktion (Mode A) aus konvertiertem QuTiP Objekt")
        plt.show()


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
    if PLOT:
        final_cv_state.plot_covariance()

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
    if PLOT:
        plot_wigner_analytically(test_state, mode_name="a", x_max=5.0)


def test_wigner_qutip_plotting():

    # 1. Setup: Verschränkten EPR-Zustand im CV-Circuit erzeugen
    circuit = GaussianCircuit()
    circuit.add_mode("a").add_mode("b")
    circuit.squeeze(mode="a", r=0.6).squeeze(
        mode="b", r=0.6, theta=np.pi
    ).beam_splitter(mode_a="a", mode_b="b", eta=0.5)

    cv_state = circuit.compile_and_run()

    # --- ANSATZ 1: Die CV-Korrelation direkt plotten ---
    if PLOT:
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
    if PLOT:
        plt.show()

    # --- B) Natives QuTiP Matrix-Histogramm (Fock-Besetzung) ---
    # Zeigt die Amplituden der Dichtematrix in der Teilchenzahl-Basis an [source: 1.1.1, 1.1.6]
    # Da es sich um ein 2-Moden-System handelt, reduzieren wir es auf Mode A für bessere Lesbarkeit
    fig, ax = matrix_histogram(rho_a)
    ax.view_init(azim=-30, elev=40)
    plt.title("Natives QuTiP Matrix-Histogramm der Mode A")
    if PLOT:
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
    if PLOT:
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
    if PLOT:
        plt.show()


def test_laser_pulse():
    # =====================================================================
    # 1. INITIALISIERUNG (Gaußscher Startzustand via venv-Circuit)
    # =====================================================================
    circuit = GaussianCircuit()
    circuit.add_mode("c")
    circuit.squeeze(mode="c", r=0.5, theta=0.0)
    # Wir starten dieses Mal im reinen Vakuum, um zu sehen, wie der Laser die Kavität füllt!
    initial_cv_state = circuit.compile_and_run()

    N_cutoff = 30  # Höherer Cutoff, da der Laser viele Photonen reinpumpt!
    rho_0 = initial_cv_state.to_qutip(N_cutoff=N_cutoff)

    # =====================================================================
    # 2. DEFINITION DES ZEITABHÄNGIGEN LASERPULSES
    # =====================================================================
    # Parameter für den Gaußschen Laserpuls
    pulse_amplitude = 2.5  # Maximale Stärke des Pulses (Rabi-Frequenz)
    pulse_center = 4.0  # Peak des Pulses bei t = 4.0
    pulse_width = 1.0  # Zeitliche Breite (Standardabweichung sigma)

    # Python-Funktion für die zeitabhängige Amplitude Omega(t) [source: 1.2.7]
    def laser_pulse_shape(t, args):
        amp = args["amplitude"]
        t0 = args["center"]
        sigma = args["width"]
        return amp * np.exp(-((t - t0) ** 2) / (2 * sigma**2))

    # =====================================================================
    # 3. HAMILTONOPERATOR & MULTI-KOMPONENTEN SIMULATION
    # =====================================================================
    omega_c = 0.0  # Wir arbeiten im rotierenden Bezugssystem des Lasers (Resonanz)
    kappa = 0.4  # Verlustrate der Kavität (Photonen-Dämpfung) [source: 1.3.1]

    a = qt.destroy(N_cutoff)

    # Statischer Teil des Hamiltonoperators (freie Energie der Mode)
    H_0 = omega_c * a.dag() * a

    # Zeitabhängiger Teil: H = H_0 + Omega(t) * (a + a^dagger)
    # QuTiP-Format für zeitabhängige Operatoren: [Operator, Funktion/String] [source: 1.2.7]
    H_drive_op = a + a.dag()
    H_total = [H_0, [H_drive_op, laser_pulse_shape]]

    # Parameter-Dictionary an QuTiP übergeben [source: 1.2.7]
    pulse_args = {
        "amplitude": pulse_amplitude,
        "center": pulse_center,
        "width": pulse_width,
    }

    # Kollaps-Operatoren für die Verlust-Dynamik [source: 1.1.4, 1.3.1]
    c_ops = [np.sqrt(kappa) * a]

    # Zeitgitter definieren (Von t=0 bis t=12)
    tlist = np.linspace(0, 12, 300)

    print("\n⚡ Starte zeitabhängige Laserpuls-Simulation (Master Equation)...")
    # Wichtig: Wir tracken die mittlere Photonenzahl (a.dag()*a) und die Quadratur <x> [source: 1.2.3]
    x_op = (a + a.dag()) / np.sqrt(2)
    result = qt.mesolve(H_total, rho_0, tlist, c_ops=c_ops, args=pulse_args)
    # e_ops=[a.dag() * a, x_op], args=pulse_args)

    photon_numbers = [qt.expect(a.dag() * a, state) for state in result.states]
    x_pect = [qt.expect(x_op, state) for state in result.states]

    # =====================================================================
    # 4. VISUALISIERUNG DER PULS-DYNAMIK
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pulsform vs. Photonenanzahl in der Kavität
    pulse_values = [laser_pulse_shape(t, pulse_args) for t in tlist]
    axes[0].plot(
        tlist,
        pulse_values,
        label=r"Laserpuls-Einhüllende $\Omega(t)$",
        color="orange",
        lw=2,
        ls="--",
    )
    axes[0].plot(
        tlist,
        photon_numbers,
        label=r"Kavitäts-Besetzung $\langle n(t) \rangle$",
        color="darkblue",
        lw=2.5,
    )
    axes[0].set_title("Einkopplung des Laserpulses in die Kavität")
    axes[0].set_xlabel("Zeit $t$")
    axes[0].set_ylabel("Amplitude / Photonenzahl")
    axes[0].grid(True, ls="--")
    axes[0].legend()

    # Plot 2: Phasenraum-Trajektorie des Displacements <x(t)>
    # Da wir eine reine Verschiebung treiben, bewegt sich der Zustand im Phasenraum
    axes[1].plot(
        tlist,
        x_pect,
        label=r"Feldamplitude $\langle x(t) \rangle$",
        color="crimson",
        lw=2,
    )
    axes[1].set_title("Reaktion der Feld-Quadratur $\\langle x \\rangle$")
    axes[1].set_xlabel("Zeit $t$")
    axes[1].set_ylabel("Erwartungswert")
    axes[1].grid(True, ls="--")
    axes[1].legend()

    plt.tight_layout()
    if PLOT:
        plt.show()


def test_kerr_state():

    # =====================================================================
    # 1. SETUP: Start im Vakuum (Hilbertraum)
    # =====================================================================
    N_cutoff = (
        35  # Höherer Cutoff wichtig, da Kerr-Zustände breite Fock-Verteilungen haben!
    )
    rho_0 = qt.ket2dm(qt.fock(N_cutoff, 0))  # Start im reinen Vakuum

    a = qt.destroy(N_cutoff)

    # =====================================================================
    # 2. DEFINITION DER KERR-DYNAMIK & LASERPULS
    # =====================================================================
    K = 0.5  # Stärke der Kerr-Nichtlinearität (massiver Effekt)
    pulse_amplitude = 4.0  # Laser-Stärke

    # Laser-Pulsform: Ein schneller, starker Puls zu Beginn, um die Kavität zu laden
    def pulse_shape(t, args):
        return args["amp"] * np.exp(-((t - args["t0"]) ** 2) / (2 * args["sigma"] ** 2))

    # Konstruktion des zeitabhängigen Hamiltonoperators [source: 1.3.8]
    # H = H_Kerr + Omega(t) * (a + a^dagger)
    H_kerr = K * a.dag() * a.dag() * a * a  # Statischer Kerr-Term [source: 1.2.1]
    H_drive = a + a.dag()  # Zeitabhängiger Laser-Treiber

    H_total = [H_kerr, [H_drive, pulse_shape]]
    pulse_args = {"amp": pulse_amplitude, "t0": 2.0, "sigma": 0.8}

    # Minimale Kavitäten-Dämpfung (Katzen-Zustände sind extrem empfindlich gegen Verlust!) [source: 1.2.5, 1.2.8]
    kappa = 0.02
    c_ops = [np.sqrt(kappa) * a]

    # Zeitgitter: Wir simulieren bis t = 6.0, um das Laden und die Kerr-Evolution zu sehen
    tlist = np.linspace(0, 6, 200)

    print("🚀 Simuliere Kerr-induzierte Schrödinger-Katzen-Präparation...")
    result = qt.mesolve(H_total, rho_0, tlist, c_ops=c_ops, args=pulse_args)

    # Wir greifen uns drei markante Zeitpunkte aus der Evolution ab
    # 1. Nach dem Puls (Zustand ist geladen und kohärent)
    # 2. Mittendrin (Squeezing und Verbiegung)
    # 3. Das mathematische Kerr-Cat-Maximum (Selbst-Interferenz)
    rho_t1 = result.states[60]  # t ~ 1.8 (kurz nach dem Puls)
    rho_t2 = result.states[110]  # t ~ 3.3 (Kerr fängt an zu verzerren)
    rho_t3 = result.states[-1]  # t = 6.0 (Kollision -> Katze!)

    # =====================================================================
    # 3. MULTI-PANEL PLOT DER KERR-EVOLUTION
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    xvec = np.linspace(-5, 5, 200)

    # Zeitschritt 1: Der Laser hat ein verschobenes "Gauß-Blob" (Coherent State) erzeugt
    W1 = qt.wigner(rho_t1, xvec, xvec)
    axes[0].contourf(xvec, xvec, W1, 100, cmap="RdBu_r")
    axes[0].set_title(r"t = 1.8: Kohärenter Laser-Puls")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("p")
    axes[0].axis("equal")

    # Zeitschritt 2: Die Kerr-Nichtlinearität verbiegt den Zustand zu einer Banane
    W2 = qt.wigner(rho_t2, xvec, xvec)
    axes[1].contourf(xvec, xvec, W2, 100, cmap="RdBu_r")
    axes[1].set_title(r"t = 3.3: Kerr-Squeezing & Verbiegung")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p")
    axes[1].axis("equal")

    # Zeitschritt 3: DIE SCHRÖDINGER KATZE (Interferenz im Phasenraum)
    W3 = qt.wigner(rho_t3, xvec, xvec)
    # vmin/vmax symmetrisch für die Quanten-Interferenzstreifen
    cont = axes[2].contourf(xvec, xvec, W3, 100, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
    fig.colorbar(cont, ax=axes[2], label="Wigner-Dichte")
    axes[2].set_title(r"t = 6.0: Kerr-Cat State")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("p")
    axes[2].axis("equal")

    plt.tight_layout()
    if PLOT:
        plt.show()


def test_cat_in_mzi():

    # =====================================================================
    # 1. PREPARATION: Den exakten Kerr-Cat-Zustand erzeugen
    # =====================================================================
    # (Wir nehmen die Katze aus unserem vorherigen Schritt direkt im Hilbertraum auf)
    N_cutoff = 22  # Dimension pro Mode im Interferometer
    a = qt.destroy(N_cutoff)

    # Wir bauen uns eine reine, ungedämpfte Katze analytisch im Hilbertraum,
    # um numerisch sauber und schnell zu bleiben: |cat> = N * (|alpha> + |-alpha>)
    alpha = 2
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    # =====================================================================
    # 2. DAS ZWEI-MODEN-MZI IM HILBERTRAUM DEFINIEREN
    # =====================================================================
    # Wir haben zwei Pfade im Interferometer: Mode 1 (Arm A) und Mode 2 (Arm B)
    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    # Der 50:50 Strahlteiler-Operator zwischen den beiden Armen
    # U_BS = exp(i * pi/4 * (a1^dagger * a2 + a1 * a2^dagger))
    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    # Eingangs-Zustand des MZI: Katze auf Port 1, Vakuum auf Port 2
    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))

    # --- SCHRITT A: Erster Strahlteiler (BS1) ---
    # Erzeugt massive Verschränkung zwischen den beiden Pfaden!
    psi_after_BS1 = U_BS * psi_in

    # --- SCHRITT B: Phasenverschiebung theta im oberen Arm (Arm 1) ---
    # Wir wählen eine Verschiebung von theta = pi/2 (90 Grad) für maximale Interferenzänderung
    theta = np.pi / 4
    U_phase = (1j * theta * a1.dag() * a1).expm()
    psi_after_phase = U_phase * psi_after_BS1

    # --- SCHRITT C: Zweiter Strahlteiler (BS2) ---
    # Rekombination der Pfade
    psi_out = U_BS * psi_after_phase

    # =====================================================================
    # 3. AUSWERTUNG & PARTIAL TRACE AN DEN AUSGANGS-PORTS
    # =====================================================================
    # Wir schauen uns an, was aus Ausgangsport 1 und Ausgangsport 2 herauskommt
    rho_out_port1 = qt.ptrace(psi_out, 0)
    rho_out_port2 = qt.ptrace(psi_out, 1)

    # =====================================================================
    # 4. VISUALISIERUNG DER INTERFERENZ
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xvec = np.linspace(-4, 4, 200)

    # Port 1 Wigner-Funktion
    W_port1 = qt.wigner(rho_out_port1, xvec, xvec)
    axes[0].contourf(xvec, xvec, W_port1, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[0].set_title("MZI Ausgangsport 1\n(Verschobene Katze bei $\\theta=\\pi/2$)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("p")
    axes[0].axis("equal")

    # Port 2 Wigner-Funktion
    W_port2 = qt.wigner(rho_out_port2, xvec, xvec)
    axes[1].contourf(xvec, xvec, W_port2, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[1].set_title("MZI Ausgangsport 2\n(Gegenphasige Interferenz)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p")
    axes[1].axis("equal")

    plt.tight_layout()
    if PLOT:
        plt.show()


def test_time_cat_mzi():

    import matplotlib.pyplot as plt
    import numpy as np
    import qutip as qt

    # =====================================================================
    # 1. SETUP: Katze und Operatoren im Hilbertraum definieren
    # =====================================================================
    N_cutoff = 22  # Hilbertraum-Dimension pro Mode
    a = qt.destroy(N_cutoff)

    # Eine reine Katze vorbereiten: |cat> = N * (|alpha> + |-alpha>)
    alpha = 4.0 + 2j
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    # Zwei-Moden-Operatoren im Gesamtraum aufbauen
    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    # Teilchenzahl-Operatoren für die Ausgänge
    n1_op = a1.dag() * a1
    n2_op = a2.dag() * a2

    # Paritäts-Operator für Port 1: P = exp(i * pi * a1^dagger * a1)
    # Er misst, ob die Photonenzahl gerade (+1) oder ungerade (-1) ist
    parity1_op = (1j * np.pi * n1_op).expm()

    # Der 50:50 Strahlteiler-Operator (BS)
    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    # Eingangszustand: Katze auf Port 1, Vakuum auf Port 2
    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))

    # =====================================================================
    # 2. PHASENSCHLEIFE (Der Phasen-Scan von 0 bis 2*pi)
    # =====================================================================
    theta_list = np.linspace(0, 2 * np.pi, 200)

    # Listen für die Messergebnisse
    mean_n1 = []
    mean_n2 = []
    parity_port1 = []

    # Erster Strahlteiler ist fix für alle Phasen (Verschränkung im MZI)
    psi_after_BS1 = U_BS * psi_in

    print("🔄 Scanne Phase theta von 0 bis 2pi...")
    for theta in theta_list:
        # 1. Phasenverschiebung im oberen Arm anwenden
        U_phase = (1j * theta * a1.dag() * a1).expm()
        psi_after_phase = U_phase * psi_after_BS1

        # 2. Zweiter Strahlteiler (Rekombination)
        psi_out = U_BS * psi_after_phase

        # 3. Erwartungswerte für diesen Phasenwert berechnen [source: 1.2.3]
        mean_n1.append(qt.expect(n1_op, psi_out))
        mean_n2.append(qt.expect(n2_op, psi_out))
        parity_port1.append(qt.expect(parity1_op, psi_out).real)

    print("✅ Scan beendet. Generiere Diagramme...")

    # =====================================================================
    # 3. VISUALISIERUNG DER INTERFERENZ-FRANSEN
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot A: Klassische Intensitäten (Photonenzahlen) an den Ausgängen
    ax1.plot(
        theta_list / np.pi, mean_n1, label="Ausgangs-Port 1", color="darkblue", lw=2
    )
    ax1.plot(
        theta_list / np.pi,
        mean_n2,
        label="Ausgangs-Port 2",
        color="crimson",
        lw=2,
        ls="--",
    )
    ax1.set_ylabel(r"Mittlere Photonenzahl $\langle n \rangle$")
    ax1.set_title("Mach-Zehnder Interferenz-Fransen (Intensität)")
    ax1.grid(True, ls="--")
    ax1.legend()

    # Plot B: Quanten-Parität am Ausgang 1
    # Das zeigt die Verschiebung der mikroskopischen Interferenzstreifen
    ax2.plot(
        theta_list / np.pi, parity_port1, label="Parität Port 1", color="purple", lw=2.5
    )
    ax2.axhline(0, color="black", lw=0.5, ls="-")
    ax2.set_xlabel(r"Phasenverschiebung $\theta$ ($\times \pi$)")
    ax2.set_ylabel("Erwartungswert der Parität")
    ax2.set_title("Quanten-Paritäts-Oszillation (Super-Auflösung)")
    ax2.grid(True, ls="--")
    ax2.legend()

    plt.tight_layout()
    if PLOT:
        plt.show()


def test_decoherence_mzi():
    start_time = perf_counter()

    # =====================================================================
    # 1. SETUP: Zustände und MZI-Operatoren im Hilbertraum
    # =====================================================================
    N_cutoff = 12
    a = qt.destroy(N_cutoff)

    # Eine reine Katze vorbereiten: |cat> = N * (|alpha> + |-alpha>)
    alpha = 2.0
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    # Zwei-Moden-Operatoren im MZI
    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    n1_op = a1.dag() * a1
    parity1_op = (1j * np.pi * n1_op).expm()  # Paritäts-Messoperator

    # Der 50:50 Strahlteiler (BS)
    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    # Eingangszustand: Katze auf Port 1, Vakuum auf Port 2
    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))
    psi_after_BS1 = U_BS * psi_in

    # =====================================================================
    # 2. PHASENSCHLEIFE MIT/OHNE VERLUST
    # =====================================================================
    divisions = 120
    theta_list = np.linspace(0, 2 * np.pi, divisions)

    parity_perfect = []
    parity_noisy = []

    # Verlustrate im oberen MZI-Arm definieren (Dämpfung)
    kappa = 0.25
    c_ops = [np.sqrt(kappa) * a1]  # Verlust wirkt NUR auf Arm 1 [1]

    print("🔄 Scanne Phase theta und berechne Dekohärenz-Effekte...")
    for theta in theta_list:

        # --- SZENARIO A: Perfektes MZI (Verlustfrei) ---
        U_phase = (1j * theta * a1.dag() * a1).expm()
        psi_out_perfect = U_BS * U_phase * psi_after_BS1
        parity_perfect.append(qt.expect(parity1_op, psi_out_perfect).real)

        # --- SZENARIO B: Noisy MZI (Zeitabhängiger Verlust via Master Equation) ---
        # Die Phase 'theta' wird hier als effektive Zeitentwicklung simuliert.
        # H = a1^dagger * a1 sorgt für die Phasenverschiebung über die Dauer t = theta
        H_phase = a1.dag() * a1

        # Wir lösen die Dynamik IM ARM für die Dauer t = theta unter Verlusten [1]
        t_span = [0, theta] if theta > 0 else [0, 1e-9]  # Nullzeit-Abfang

        # sim_res = qt.mesolve(H_phase, psi_after_BS1, t_span, c_ops=c_ops) [1]
        sim_res = qt.mesolve(H_phase, psi_after_BS1, t_span, c_ops=c_ops)
        # sim_res = sim.states[1]
        rho_after_arm_with_loss = sim_res.states[-1]  # Zustand nach dem Arm

        # Rekombination am zweiten Strahlteiler
        rho_out_noisy = U_BS * rho_after_arm_with_loss * U_BS.dag()
        parity_noisy.append(qt.expect(parity1_op, rho_out_noisy).real)

    stop_time = perf_counter()
    print(f"✅ Simulation beendet, Laufzeit: [{stop_time - start_time}]\n"
          f"Parameter: <{N_cutoff=}> \t <{divisions=}>"
          )


    # =====================================================================
    # 3. PLOT: Perfekte vs. Zerstörte Quanten-Interferenz
    # =====================================================================
    plt.figure(figsize=(10, 5))
    plt.plot(
        theta_list / np.pi,
        parity_perfect,
        label="Perfektes MZI (Kein Verlust)",
        color="purple",
        lw=2,
    )
    plt.plot(
        theta_list / np.pi,
        parity_noisy,
        label=r"Verlustbehaftetes MZI ($\kappa = 0.25$)",
        color="crimson",
        lw=2.5,
        ls="--",
    )

    plt.axhline(0, color="black", lw=0.5, ls="-")
    plt.xlabel(r"Phasenverschiebung $\theta$ ($\times \pi$)")
    plt.ylabel("Erwartungswert der Parität")
    plt.title("Dekohärenz im QBS-Simulator: Kollaps der Katzen-Interferenz")
    plt.grid(True, ls="--")
    plt.legend()
    plt.tight_layout()
    if PLOT:
        plt.show()


def test_triggered_cavity():
    # 1. Schnelle CV-Vorpräparation im Vakuum
    cv_circuit = GaussianCircuit().add_mode("c")
    # cv_circuit.add_mode("d")
    cv_circuit.squeeze(mode="c", r=0.1, theta=0.0)
    initial_state = cv_circuit.compile_and_run()

    # 2. Hybrid-Schnittstelle zünden & in die Fock-Basis wechseln
    N_fock = 15
    # rho_vacuum = initial_state.to_qutip(N_cutoff=N_fock)

    rho_vacuum = initial_state.to_qutip(N_cutoff=N_fock)

    # 3. Zeitabhängigen Laser-Puls + Kerr-Effekt in der Kavität simulieren
    tlist = np.linspace(0, 5, 100)
    print("Simuliere physikalische Echtzeit-Dynamik...")
    states = QBSSimulator.run_cavity_with_pulse(
        rho_init=rho_vacuum,
        tlist=tlist,
        K=0.4,
        kappa=0.05,
        amp=3.5,
        t0=1.5,
        sigma=0.6,
        N_cutoff=N_fock,
    )
    rho_kerr_cat = states[-1]
    print(f"{rho_kerr_cat.shape=}")

    # 4. Nicht-Gaußschen Photonen-Abzug nach dem Kavitätsaustritt triggern
    rho_final_non_gaussian = QBSSimulator.photon_subtraction(
        rho_kerr_cat, N_cutoff=N_fock
    )

    # Sanity Check über die native Quanten-Reinheit (Purity) tr(rho^2)
    purity = (rho_final_non_gaussian * rho_final_non_gaussian).tr().real
    print(
        f"✅ Integrationstest bestanden! Quantenreinheit des Endzustands: {purity:.4f}"
    )

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from .states import (
    GaussianCircuit,
    GaussianMeasurements,
    GaussianOperations,
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


def test_wigner_qutip_plotting():
    import matplotlib.pyplot as plt
    import qutip as qt
    from qutip.visualization import matrix_histogram

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

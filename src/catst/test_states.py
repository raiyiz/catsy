import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from .states import GaussianOperations, QBSChannels


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
    state = GaussianOperations.apply_beam_splitter(state, mode_a="a", mode_b="b", eta=0.5)

    print("\nKovarianz V nach dem Beam Splitter (Verschränkung ist entstanden!):")
    print(np.round(state.covariance, 3))

    # 4. Plotten der resultierenden Kovarianzmatrix
    state.plot_covariance(do_plot=False)

def test_cv_chan_to_fock():

    # 1. Erzeuge einen verschränkten Zustand mit Rauschen im CV-Formalismus
    state = GaussianOperations.create_vacuum(modes=("a", "b"))
    state = GaussianOperations.apply_squeezing(state, mode="a", r=0.5)
    state = GaussianOperations.apply_squeezing(state, mode="b", r=0.5, theta=np.pi/2)
    state = GaussianOperations.apply_beam_splitter(state, mode_a="a", mode_b="b", eta=0.5)
    
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
    plt.show()

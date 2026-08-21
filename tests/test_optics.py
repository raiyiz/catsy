from time import perf_counter

import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.fock import FockOperations
from catsy.gaussian import (
    GaussianCircuit,
    GaussianState,
    beam_splitter,
    compute_duan_inseparability,
    loss,
    rotate,
    squeeze,
)
from catsy.optics import (
    KerrCavity,
    MachZehnderInterferometer,
    OpticalComponent,
    OpticalSetup,
)

# Layout assembly and serialization


def test_beam_splitter_registers_both_ports_and_populates_circuit():
    setup = OpticalSetup("Bench").beam_splitter("BS1", port_a="a", port_b="b", eta=0.5)
    assert setup.registered_ports == {"a", "b"}
    assert setup.circuit.modes == ("a", "b")
    assert setup.circuit.to_dict()["operations"] == [
        {"name": "beam_splitter", "modes": ["a", "b"], "kwargs": {"eta": 0.5}}
    ]
    assert setup.components[0].op_type == "beam_splitter"
    assert setup.components[0].kwargs == {"eta": 0.5}


def test_optical_setup_uses_injected_circuit():
    circuit = GaussianCircuit().add_mode("a")
    setup = OpticalSetup("Bench", circuit=circuit).phase_shifter(
        "Phase", port="a", phi=0.25
    )

    assert setup.circuit is circuit
    assert setup.circuit.to_dict()["operations"] == [
        {"name": "rotate", "modes": ["a"], "kwargs": {"phi": 0.25}}
    ]


def test_process_beam_does_not_rebuild_or_duplicate_circuit_operations():
    setup = OpticalSetup("Bench").phase_shifter("Phase", port="a", phi=0.25)
    input_state = GaussianState.coherent(modes=("a",), alphas=[1.0])

    first = setup.process_beam(input_state)
    second = setup.process_beam(input_state)

    assert np.allclose(first.displacement, second.displacement)
    assert len(setup.circuit.to_dict()["operations"]) == 1


def test_layout_roundtrips_through_file(tmp_path):
    setup = OpticalSetup("MZI Node")
    setup.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    setup.fiber_loss("Loss_A", port="line_1", eta=0.9)
    setup.phase_shifter("Phase_B", port="line_2", phi=0.785)
    setup.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    layout_path = tmp_path / "mzi_node.json"
    setup.save_layout(layout_path)
    loaded = OpticalSetup.load_layout(layout_path)

    assert loaded.name == "MZI Node"
    assert [c.name for c in loaded.components] == ["BS1", "Loss_A", "Phase_B", "BS2"]
    assert loaded.registered_ports == {"line_1", "line_2"}
    assert loaded.components[2].kwargs == {"phi": 0.785}


def test_component_to_dict_and_from_dict_agree():
    comp = OpticalComponent("BS1", beam_splitter, ("a", "b"), {"eta": 0.5})
    assert OpticalComponent.from_dict(comp.to_dict()) == comp


def test_component_owns_the_executable_callable():
    comp = OpticalComponent("BS1", beam_splitter, ("a", "b"), {"eta": 0.5})

    assert comp.op is beam_splitter
    assert comp.op_type == "beam_splitter"
    assert comp.ports == ("a", "b")
    assert comp.kwargs == {"eta": 0.5}


def test_optical_component_serializes_the_bare_function_name():
    comp = OpticalComponent("BS1", beam_splitter, ("a", "b"), {"eta": 0.5})

    assert comp.to_dict() == {
        "name": "BS1",
        "op_type": "beam_splitter",
        "ports": ["a", "b"],
        "kwargs": {"eta": 0.5},
    }
    assert OpticalComponent.from_dict(comp.to_dict()).op is beam_splitter


@pytest.mark.parametrize("op", [beam_splitter, loss, squeeze, rotate])
def test_optical_component_accepts_only_known_optical_callables(op):
    component = {
        beam_splitter: OpticalComponent("BS", beam_splitter, ("a", "b"), {"eta": 0.5}),
        loss: OpticalComponent("Loss", loss, ("a",), {"eta": 0.9}),
        squeeze: OpticalComponent("Sqz", squeeze, ("a",), {"r": 0.5, "theta": 0.0}),
        rotate: OpticalComponent("Phase", rotate, ("a",), {"phi": 0.2}),
    }[op]
    assert component.op is op


def test_optical_component_rejects_unknown_callable():
    def custom_operation(state, modes, **kwargs):
        return state

    with pytest.raises(ValueError, match="Unknown optical component operation"):
        OpticalComponent("Custom", custom_operation, ("a",), {})


# Execution


def test_process_beam_rejects_empty_setup():
    setup = OpticalSetup("Empty Bench")
    vacuum = GaussianState.vacuum(modes=("a",))
    with pytest.raises(ValueError):
        setup.process_beam(vacuum)


def test_mzi_setup_preserves_purity_for_coherent_input():
    mzi = OpticalSetup("MZI Node")
    mzi.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    mzi.phase_shifter("Phase", port="line_2", phi=0.785)
    mzi.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    coherent_in = GaussianState.coherent(
        modes=("line_1", "line_2"), alphas=[1.5 + 0.0j, 2.0j]
    )
    result = mzi.process_beam(coherent_in)
    # A lossless 2-mode MZI (no fiber_loss component) is purity-preserving:
    # det(V) stays at the pure-state value of 0.5**(2*n_modes) == 0.0625.
    assert np.linalg.det(result.covariance) == pytest.approx(0.0625, rel=1e-9)


def test_lossy_channel_setup_weakens_but_can_preserve_entanglement():
    setup = OpticalSetup("Lossy Channel")
    setup.fiber_loss("Loss_A", port="line_1", eta=0.9)
    setup.fiber_loss("Loss_B", port="line_2", eta=1.0)  # lossless reference arm

    tmsv_in = GaussianState.tmsv(mode_a="line_1", mode_b="line_2", r=1.2)
    duan_before = compute_duan_inseparability(tmsv_in, "line_1", "line_2")

    result = setup.process_beam(tmsv_in)
    duan_after = compute_duan_inseparability(result, "line_1", "line_2")

    # Loss always moves the witness toward the separability bound...
    assert duan_after > duan_before
    # ...but 10% loss on one arm isn't enough to destroy r=1.2 entanglement.
    assert duan_after < 2.0


def test_process_beam_rejects_input_state_missing_a_registered_port():
    setup = OpticalSetup("Bench").beam_splitter("BS1", port_a="a", port_b="b", eta=0.5)
    mismatched = GaussianState.vacuum(modes=("a",))
    with pytest.raises(ValueError):
        setup.process_beam(mismatched)


# Schematic rendering


def test_render_schematic_of_empty_setup():
    schematic = OpticalSetup("Empty Bench").render_schematic()
    assert "Empty Bench Layout" in schematic


def test_render_schematic_labels_each_component_and_input_state():
    setup = OpticalSetup("MZI Node")
    setup.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    setup.fiber_loss("Loss_A", port="line_1", eta=0.9)
    setup.phase_shifter("Phase_B", port="line_2", phi=0.785)

    schematic = setup.render_schematic(
        input_states={"line_1": "|a=1.5>", "line_2": "|b=0.8>"}
    )
    assert "|a=1.5>" in schematic
    assert "|b=0.8>" in schematic
    assert "beam" in schematic
    assert "loss" in schematic
    assert "rotat" in schematic
    assert "line_1" in schematic and "line_2" in schematic
    # print(schematic)


def test_render_schematic_bridges_ports_a_multi_port_component_skips_over():
    # 3 ports, but the beam splitter only touches the outer two -- the
    # middle port ("b") must be bridged, not silently dropped or crashed on
    # (this is the involved_ports[0] vs involved_ports comparison bug).
    setup = OpticalSetup("Bridge Test")
    setup.beam_splitter("BS1", port_a="a", port_b="c", eta=0.5)
    setup.phase_shifter("Phase", port="b", phi=0.1)

    schematic = setup.render_schematic()
    lines = schematic.splitlines()
    b_line = next(line for line in lines if "[b]" in line)
    assert "│" in b_line


def test_draw_prints_the_rendered_schematic(capsys):
    setup = OpticalSetup("Bench").beam_splitter("BS1", port_a="a", port_b="b", eta=0.5)
    setup.draw()
    captured = capsys.readouterr()
    assert captured.out.strip() == setup.render_schematic().strip()


@pytest.mark.visual
def test_visual_schematic_draw():
    # Schema one
    mzi = OpticalSetup("MZI Interferometer Node")
    mzi.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    mzi.fiber_loss("Loss_A", port="line_1", eta=0.9)
    mzi.phase_shifter("Phase_B", port="line_2", phi=0.785)
    mzi.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    mzi.draw(input_states={"Route 1": "|α=1.5>", "Route 2": "|ξ=0.8>"})

    # Schema two
    mzi = OpticalSetup("Colored MZI Architecture")
    mzi.inline_squeezer("Sqz_Input", port="line_2", r=0.7)
    mzi.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    mzi.fiber_loss("Loss_A", port="line_1", eta=0.85)
    mzi.phase_shifter("Phase_B", port="line_2", phi=1.57)
    mzi.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    mzi.draw(input_states={"Route 1": "|α=2.0>", "Route 2": "|0>"})


# Cavity and interferometer visual diagnostics


@pytest.mark.visual
def test_laser_pulse_cavity_plot_demo():
    import matplotlib.pyplot as plt

    N_cutoff = 15
    rho_vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    tlist = np.linspace(0, 5, 100)
    states = KerrCavity(K=0.3, kappa=0.05, N_cutoff=N_cutoff).run(
        rho_init=rho_vacuum,
        tlist=tlist,
        amp=3.0,
        t0=1.5,
        sigma=0.5,
    )
    n_op = qt.num(N_cutoff)
    photon_numbers = [qt.expect(n_op, s) for s in states]

    plt.figure(figsize=(6, 4))
    plt.plot(tlist, photon_numbers)
    plt.xlabel("time")
    plt.ylabel("<n>")
    plt.title("Driven Kerr cavity: photon number vs time")
    plt.show()


@pytest.mark.visual
def test_full_cavity_multipanel_plot_demo():
    import matplotlib.pyplot as plt

    N_cutoff = 12
    alpha = 1.5
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.linspace(0, 2 * np.pi, 60)
    results = MachZehnderInterferometer(kappa=0.2, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )

    _fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(theta_list, results["n1"])
    axes[0].set_title("Mean photon number, arm 1")
    axes[1].plot(theta_list, results["n2"])
    axes[1].set_title("Mean photon number, arm 2")
    axes[2].plot(theta_list, results["parity1"])
    axes[2].set_title("Parity, arm 1")
    plt.tight_layout()
    plt.show()


# Cavity and interferometer simulations


def test_triggered_cavity_end_to_end():
    cv_circuit = GaussianCircuit().add_mode("c")
    cv_circuit.squeeze(mode="c", r=0.1, theta=0.0)
    initial_state = cv_circuit.compile_and_run()

    N_fock = 15
    rho_vacuum = initial_state.to_qutip(N_cutoff=N_fock)

    tlist = np.linspace(0, 5, 60)
    states = KerrCavity(K=0.4, kappa=0.05, N_cutoff=N_fock).run(
        rho_init=rho_vacuum,
        tlist=tlist,
        amp=3.5,
        t0=1.5,
        sigma=0.6,
    )
    assert len(states) == len(tlist)
    rho_kerr_cat = states[-1]
    assert rho_kerr_cat.tr() == pytest.approx(1.0, abs=1e-6)

    rho_final_non_gaussian = FockOperations.photon_subtraction(
        rho_kerr_cat, N_cutoff=N_fock
    )
    purity = (rho_final_non_gaussian * rho_final_non_gaussian).tr().real
    assert 0.0 < purity <= 1.0 + 1e-9


def test_decoherence_mzi_parity_visibility_drops_with_loss():
    start_time = perf_counter()
    N_cutoff = 10
    alpha = 1.5
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    # Loss has a fixed exposure time and is therefore independent of the
    # scanned phase.  Compare the complete phase scan directly.
    theta_list = np.linspace(0, 2 * np.pi, 80)
    results_clean = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )
    results_noisy = MachZehnderInterferometer(kappa=0.4, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )

    tail = slice(len(theta_list) // 2, None)
    visibility_clean = np.ptp(np.array(results_clean["parity1"])[tail])
    visibility_noisy = np.ptp(np.array(results_noisy["parity1"])[tail])
    # Loss must wash out (not enhance) the super-resolved parity fringes.
    assert visibility_noisy < visibility_clean
    print(f"MZI decoherence scan runtime: {perf_counter() - start_time:.2f}s")


def test_mzi_phase_scan_is_independent_of_loss_when_exposure_time_is_zero():
    N_cutoff = 10
    alpha = 1.2
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.array([-0.7, 0.0, 0.9])

    clean = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )
    zero_exposure = MachZehnderInterferometer(
        kappa=10.0, N_cutoff=N_cutoff, loss_time=0.0
    ).scan(psi_cat, theta_list)

    for key in ("n1", "n2", "parity1"):
        np.testing.assert_allclose(zero_exposure[key], clean[key], atol=1e-10, rtol=1e-10)


def test_mzi_negative_phase_is_not_clipped_to_zero():
    N_cutoff = 12
    alpha = 1.0
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    result = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff).scan(
        psi_cat, np.array([-0.8, 0.0, 0.8])
    )

    # A real phase scan must distinguish a negative phase from zero.
    assert not np.allclose(result["n1"][0], result["n1"][1], atol=1e-8)


# Kerr and cat-state simulations


@pytest.mark.visual
def test_kerr_cat_state_generation(plot_enabled):
    # A driven, weakly-damped Kerr cavity: a fast pulse loads the cavity with
    # a coherent state, then the Kerr nonlinearity shears it into a cat
    # state. Routed through KerrCavity.run rather than
    # hand-rolled here, so this test and test_triggered_cavity_end_to_end
    # exercise the exact same code path the rest of the suite relies on.
    N_cutoff = 35  # Kerr cat states have wide Fock-number support -> needs a high cutoff.
    rho_vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    tlist = np.linspace(0, 6, 200)

    states = KerrCavity(
        K=0.5,  # Kerr nonlinearity strength
        kappa=0.01,  # light damping -- cat states are fragile against loss
        N_cutoff=N_cutoff,
    ).run(
        rho_init=rho_vacuum,
        tlist=tlist,
        amp=5.0,
        t0=2.0,
        sigma=0.8,
    )
    assert len(states) == len(tlist)
    assert states[-1].tr() == pytest.approx(1.0, abs=1e-6)

    if plot_enabled:
        # Snapshots: pulse arriving, freshly-displaced coherent blob, Kerr
        # shear starting to bend it, and the final cat state.
        snapshot_indices = [10, 40, 110, -1]
        snapshot_labels = [
            "t ~ 0.3: pulse arriving",
            "t ~ 1.2: displaced coherent blob",
            "t ~ 3.3: Kerr shear setting in",
            "t = 6.0: Kerr cat state",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        xvec = np.linspace(-5, 5, 200)
        cont = None
        for ax, idx, label in zip(
            axes.flat, snapshot_indices, snapshot_labels, strict=True
        ):
            W = qt.wigner(states[idx], xvec, xvec)
            cont = ax.contourf(xvec, xvec, W, 100, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
            ax.set_title(label)
            ax.set_xlabel("x")
            ax.set_ylabel("p")
            ax.axis("equal")

        fig.colorbar(cont, ax=axes[:, :], label="Wigner density")
        plt.show()


@pytest.mark.visual
def test_cat_state_single_shot_through_mzi():
    # Single-phase companion to test_decoherence_mzi_parity_visibility_drops_with_loss
    # below: a loss-free MZI at one fixed theta, kept mainly to inspect the
    # output Wigner functions -- so it belongs with the other visual-only
    # demos and follows the same skip-when-not-plotting convention.

    N_cutoff = 22
    alpha = 2
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    # 50:50 beam splitter between the two arms: U_BS = exp(i*pi/4*(a1^dag*a2 + a1*a2^dag)).
    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    # MZI input: cat state on port 1, vacuum on port 2.
    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))

    # First beam splitter entangles the two arms.
    psi_after_bs1 = U_BS * psi_in

    # Phase shift theta in the upper arm (arm 1).
    theta = np.pi / 4
    U_phase = (1j * theta * a1.dag() * a1).expm()
    psi_after_phase = U_phase * psi_after_bs1

    # Second beam splitter recombines the arms.
    psi_out = U_BS * psi_after_phase

    rho_out_port1 = qt.ptrace(psi_out, 0)
    rho_out_port2 = qt.ptrace(psi_out, 1)
    assert rho_out_port1.tr() == pytest.approx(1.0, abs=1e-6)
    assert rho_out_port2.tr() == pytest.approx(1.0, abs=1e-6)

    _fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xvec = np.linspace(-4, 4, 200)

    W_port1 = qt.wigner(rho_out_port1, xvec, xvec)
    axes[0].contourf(xvec, xvec, W_port1, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[0].set_title(r"MZI output port 1 (shifted cat, $\theta=\pi/4$)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("p")
    axes[0].axis("equal")

    W_port2 = qt.wigner(rho_out_port2, xvec, xvec)
    axes[1].contourf(xvec, xvec, W_port2, 100, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    axes[1].set_title("MZI output port 2 (out-of-phase interference)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p")
    axes[1].axis("equal")

    plt.tight_layout()
    plt.show()


@pytest.mark.visual
def test_cat_mzi_phase_scan_fringes():
    # Loss-free (kappa=0) phase scan of a cat state through the MZI -- the
    # clean-case counterpart plotted alongside the noisy one in
    # test_decoherence_mzi_parity_visibility_drops_with_loss. Routed through
    # MachZehnderInterferometer.scan instead of duplicating that loop here,
    # so both tests exercise the same simulation code.

    N_cutoff = 22
    alpha = 4.0 + 2j
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.linspace(0, 2 * np.pi, 200)

    results = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(
        theta_list / np.pi, results["n1"], label="Output port 1", color="darkblue", lw=2
    )
    ax1.plot(
        theta_list / np.pi,
        results["n2"],
        label="Output port 2",
        color="crimson",
        lw=2,
        ls="--",
    )
    ax1.set_ylabel(r"Mean photon number $\langle n \rangle$")
    ax1.set_title("Mach-Zehnder interference fringes (intensity)")
    ax1.grid(True, ls="--")
    ax1.legend()

    ax2.plot(
        theta_list / np.pi,
        results["parity1"],
        label="Parity, port 1",
        color="purple",
        lw=2.5,
    )
    ax2.axhline(0, color="black", lw=0.5, ls="-")
    ax2.set_xlabel(r"Phase shift $\theta$ ($\times \pi$)")
    ax2.set_ylabel("Parity expectation value")
    ax2.set_title("Quantum parity oscillation (super-resolution)")
    ax2.grid(True, ls="--")
    ax2.legend()

    plt.tight_layout()
    plt.show()

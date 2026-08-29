import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import qutip as qt
from matplotlib import pyplot as plt

from catsy.fock import FockGates
from catsy.gaussian import (
    GaussianState,
    beam_splitter,
    compute_duan_inseparability,
    squeeze,
)
from catsy.optics import Circuit, Gate, KerrCavity, MachZehnderInterferometer, Mode

# Layout assembly and serialization


def test_beam_splitter_builds_a_two_mode_gate():
    circuit = Circuit(name="Bench").add_mode("a").add_mode("b")
    circuit.beam_splitter("a", "b", eta=0.5)
    assert tuple(circuit.modes) == (circuit.modes.a, circuit.modes.b)
    assert circuit.to_dict()["gates"] == [
        {"gate": "BeamSplitter", "modes": ["a", "b"], "kwargs": {"eta": 0.5}}
    ]
    assert circuit.gates[0].name == "BeamSplitter"
    assert circuit.gates[0].kwargs == {"eta": 0.5}


# Mode ownership


def test_mode_returns_an_owned_handle():
    circuit = Circuit(name="Bench")
    a = circuit.mode("a")
    assert a.name == "a"
    assert a.index == 0
    assert a is circuit.modes.a
    assert a.owner is circuit


def test_mode_rejects_duplicate_name():
    circuit = Circuit().add_mode("a")
    assert circuit.mode("a") is circuit.modes.a


def test_mode_rejects_empty_name():
    circuit = Circuit()
    with pytest.raises(ValueError, match="non-empty string"):
        circuit.mode("")


def test_free_mode_has_no_owner():
    free = Mode("a")
    assert free.owner is None
    assert repr(free) == "Mode('a')"


def test_mode_uses_identity_equality():
    first = Mode("a")
    second = Mode("a")
    assert first.name == second.name
    assert first is not second
    assert first != second


def test_owned_mode_can_build_a_gate_on_its_circuit():
    circuit = Circuit(name="Bench")
    a = circuit.mode("a")
    b = circuit.mode("b")
    circuit.beam_splitter(a, b, eta=0.5)
    assert circuit.to_dict()["gates"] == [
        {"gate": "BeamSplitter", "modes": ["a", "b"], "kwargs": {"eta": 0.5}}
    ]
    assert circuit.gates[0].modes == (a, b)


def test_mode_from_another_circuit_is_rejected():
    other = Circuit(name="Other")
    foreign = other.mode("a")

    circuit = Circuit(name="Bench").add_mode("a")
    with pytest.raises(ValueError, match="belongs to circuit 'Other'"):
        circuit.squeeze(foreign, r=0.5)


def test_free_mode_is_rejected_on_a_circuit():
    free = Mode("a")
    circuit = Circuit(name="Bench").add_mode("a")
    with pytest.raises(ValueError, match="free/standalone mode"):
        circuit.squeeze(free, r=0.5)


def test_unregistered_mode_string_is_rejected_by_fluent_builder():
    circuit = Circuit(name="Bench").add_mode("a")
    with pytest.raises(ValueError, match="not registered on this circuit"):
        circuit.squeeze("z", r=0.5)


def test_owned_mode_works_with_initial_state():
    circuit = Circuit(name="Bench")
    a = circuit.mode("a")
    circuit.initial_state(a, kind="vacuum")
    final = circuit.run()
    assert final.modes == ("a",)


def test_run_does_not_rebuild_or_duplicate_circuit_gates():
    circuit = Circuit(name="Bench").add_mode("a").rotate("a", phi=0.25)
    input_state = GaussianState.coherent(modes=("a",), alphas=[1.0])

    first = circuit.run(input_state)
    second = circuit.run(input_state)

    assert np.allclose(first.displacement, second.displacement)
    assert len(circuit.to_dict()["gates"]) == 1


def test_run_rejects_a_gate_before_any_initial_state():
    circuit = Circuit(name="Bench").add_mode("a").rotate("a", phi=0.25)
    with pytest.raises(ValueError, match="cannot run before an initial_state gate"):
        circuit.run()


def test_run_requires_a_state_from_somewhere():
    circuit = Circuit(name="Bench").add_mode("a")
    with pytest.raises(ValueError, match="has to be initialized with a state"):
        circuit.run()


def test_unregistered_gate_name_raises_attribute_error():
    circuit = Circuit(name="Bench").add_mode("a")
    with pytest.raises(AttributeError):
        circuit.not_a_registered_gate("a")


def test_circuit_roundtrips_through_file(tmp_path):
    circuit = Circuit(name="MZI Node").add_mode("line_1").add_mode("line_2")
    circuit.beam_splitter("line_1", "line_2", eta=0.5)
    circuit.loss("line_1", eta=0.9)
    circuit.rotate("line_2", phi=0.785)
    circuit.beam_splitter("line_1", "line_2", eta=0.5)

    layout_path = tmp_path / "mzi_node.json"
    circuit.save(layout_path)
    loaded = Circuit.load(layout_path)

    assert loaded.name == "MZI Node"
    assert [gate.name for gate in loaded.gates] == [
        "BeamSplitter",
        "Noise",
        "Rotator",
        "BeamSplitter",
    ]
    assert {mode.name for mode in loaded.modes} == {"line_1", "line_2"}
    assert loaded.gates[2].kwargs == {"phi": 0.785}
    assert loaded.gates[0].modes == (loaded.modes.line_1, loaded.modes.line_2)


def test_circuit_gate_roundtrip_agrees(tmp_path):
    circuit = Circuit(name="Bench").add_mode("a").add_mode("b")
    circuit.beam_splitter("a", "b", eta=0.5)
    path = tmp_path / "bench.json"
    circuit.save(path)
    restored = Circuit.load(path)
    assert restored.to_dict() == circuit.to_dict()


def test_circuit_defers_physical_validation_to_gaussian_state():
    gate = Gate("BeamSplitter", beam_splitter, ("a", "b"), {"eta": 1.5})
    circuit = Circuit(name="Bench", modes=("a", "b")).add_gate(gate)
    with pytest.raises(ValueError, match="eta"):
        circuit.run(GaussianState.vacuum(("a", "b")))


def test_circuit_stores_bound_mode_objects():
    gate = Gate("BeamSplitter", beam_splitter, ("a", "b"), {"eta": 0.5})
    circuit = Circuit(name="Bench", modes=("a", "b")).add_gate(gate)
    assert circuit.gates[0].name == "BeamSplitter"
    assert circuit.gates[0].transform is beam_splitter
    assert circuit.gates[0].modes == (circuit.modes.a, circuit.modes.b)
    assert circuit.gates[0].kwargs == {"eta": 0.5}


def test_circuit_from_dict_rejects_unknown_gate():
    data = {
        "name": "Bad",
        "modes": ["a"],
        "gates": [{"gate": "not_a_real_gate", "modes": ["a"], "kwargs": {}}],
    }
    path = Path("bad_circuit.json")
    try:
        path.write_text(json.dumps(data))
        with pytest.raises(KeyError, match="not_a_real_gate"):
            Circuit.load(path)
    finally:
        path.unlink(missing_ok=True)


# Execution


def test_empty_circuit_passes_a_given_state_through_unchanged():
    circuit = Circuit(name="Empty Bench").add_mode("a")
    vacuum = GaussianState.vacuum(modes=("a",))
    result = circuit.run(vacuum)
    np.testing.assert_allclose(result.displacement, vacuum.displacement)
    np.testing.assert_allclose(result.covariance, vacuum.covariance)


def test_mzi_preserves_purity_for_coherent_input():
    mzi = Circuit(name="MZI Node").add_mode("line_1").add_mode("line_2")
    mzi.beam_splitter("line_1", "line_2", eta=0.5)
    mzi.rotate("line_2", phi=0.785)
    mzi.beam_splitter("line_1", "line_2", eta=0.5)

    coherent_in = GaussianState.coherent(
        modes=("line_1", "line_2"), alphas=[1.5 + 0.0j, 2.0j]
    )
    result = mzi.run(coherent_in)
    assert np.linalg.det(result.covariance) == pytest.approx(0.0625, rel=1e-9)


def test_lossy_channel_weakens_but_can_preserve_entanglement():
    circuit = Circuit(name="Lossy Channel").add_mode("line_1").add_mode("line_2")
    circuit.loss("line_1", eta=0.9)
    circuit.loss("line_2", eta=1.0)

    tmsv_in = GaussianState.tmsv(mode_a="line_1", mode_b="line_2", r=1.2)
    duan_before = compute_duan_inseparability(tmsv_in, "line_1", "line_2")

    result = circuit.run(tmsv_in)
    duan_after = compute_duan_inseparability(result, "line_1", "line_2")

    assert duan_after > duan_before
    assert duan_after < 2.0


def test_run_rejects_input_state_missing_a_registered_mode():
    circuit = Circuit(name="Bench", modes=("a", "b"))
    circuit.beam_splitter("a", "b", eta=0.5)
    mismatched = GaussianState.vacuum(modes=("a",))
    with pytest.raises(ValueError):
        circuit.run(mismatched)


# Schematic rendering


def test_render_schematic_of_empty_circuit():
    schematic = Circuit(name="Empty Bench").render_schematic()
    assert "Empty Bench" in schematic


def test_render_schematic_labels_each_gate_and_input_state():
    circuit = Circuit(name="MZI Node").add_mode("line_1").add_mode("line_2")
    circuit.beam_splitter("line_1", "line_2", eta=0.5)
    circuit.loss("line_1", eta=0.9)
    circuit.rotate("line_2", phi=0.785)
    circuit.squeeze("line_1", r=0.4)

    schematic = circuit.render_schematic(
        input_states={"line_1": "|a=1.5>", "line_2": "|b=0.8>"}
    )
    assert "|a=1.5>" in schematic
    assert "|b=0.8>" in schematic
    assert "BS" in schematic
    assert "LOSS" in schematic
    assert "PHASE" in schematic
    assert "SQZ" in schematic
    assert "line_1" in schematic and "line_2" in schematic


def test_render_schematic_bridges_modes_a_multi_mode_gate_skips_over():
    circuit = Circuit(name="Bridge Test").add_mode("a").add_mode("b").add_mode("c")
    circuit.beam_splitter("a", "c", eta=0.5)
    circuit.rotate("b", phi=0.1)

    schematic = circuit.render_schematic()
    lines = schematic.splitlines()
    b_line = next(line for line in lines if "[b]" in line)
    assert "│" in b_line


def test_render_schematic_labels_initial_state_with_no_parameter_suffix():
    circuit = Circuit(name="Bench").add_mode("a").initial_state("a", kind="vacuum")
    schematic = circuit.render_schematic()
    assert "INIT" in schematic


def test_draw_prints_the_rendered_schematic(capsys):
    circuit = Circuit(name="Bench", modes=("a", "b"))
    circuit.beam_splitter("a", "b", eta=0.5)
    circuit.draw()
    captured = capsys.readouterr()
    assert captured.out.strip() == circuit.render_schematic().strip()


@pytest.mark.visualize
def test_visual_schematic_draw(capsys):
    mzi = Circuit(name="MZI Interferometer Node").add_mode("line_1").add_mode("line_2")
    mzi.beam_splitter("line_1", "line_2", eta=0.5)
    mzi.loss("line_1", eta=0.9)
    mzi.rotate("line_2", phi=0.785)
    mzi.beam_splitter("line_1", "line_2", eta=0.5)

    schema_one_states = {"Route 1": "|α=1.5>", "Route 2": "|ξ=0.8>"}
    mzi.draw(input_states=schema_one_states)
    captured = capsys.readouterr()
    assert captured.out.strip() == mzi.render_schematic(schema_one_states).strip()

    mzi = Circuit(name="Colored MZI Architecture").add_mode("line_1").add_mode("line_2")
    mzi.squeeze("line_2", r=0.7)
    mzi.beam_splitter("line_1", "line_2", eta=0.5)
    mzi.loss("line_1", eta=0.85)
    mzi.rotate("line_2", phi=1.57)
    mzi.beam_splitter("line_1", "line_2", eta=0.5)

    schema_two_states = {"Route 1": "|α=2.0>", "Route 2": "|0>"}
    mzi.draw(input_states=schema_two_states)
    captured = capsys.readouterr()
    assert captured.out.strip() == mzi.render_schematic(schema_two_states).strip()


# Cavity and interferometer parameter validation


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"K": np.nan, "kappa": 0.1, "N_cutoff": 10}, "K must be finite"),
        ({"K": 0.5, "kappa": -0.1, "N_cutoff": 10}, "kappa must be finite and >= 0"),
        ({"K": 0.5, "kappa": np.inf, "N_cutoff": 10}, "kappa must be finite and >= 0"),
        ({"K": 0.5, "kappa": 0.1, "N_cutoff": 0}, "N_cutoff must be a positive integer"),
    ],
)
def test_kerr_cavity_rejects_invalid_constructor_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        KerrCavity(**kwargs)


@pytest.mark.parametrize(
    ("run_kwargs", "match"),
    [
        ({"tlist": np.array([1.0]), "amp": 1.0, "t0": 0.0, "sigma": 1.0}, "at least 2"),
        (
            {
                "tlist": np.array([[0.0, 1.0], [2.0, 3.0]]),
                "amp": 1.0,
                "t0": 0.0,
                "sigma": 1.0,
            },
            "1D array",
        ),
        (
            {"tlist": np.array([0.0, np.nan, 1.0]), "amp": 1.0, "t0": 0.0, "sigma": 1.0},
            "finite values",
        ),
        ({"tlist": np.linspace(0, 1, 5), "amp": np.nan, "t0": 0.0, "sigma": 1.0}, "amp"),
        ({"tlist": np.linspace(0, 1, 5), "amp": 1.0, "t0": np.inf, "sigma": 1.0}, "t0"),
        ({"tlist": np.linspace(0, 1, 5), "amp": 1.0, "t0": 0.0, "sigma": -1.0}, "sigma"),
        (
            {"tlist": np.linspace(0, 1, 5), "amp": 1.0, "t0": 0.0, "sigma": np.nan},
            "sigma",
        ),
    ],
)
def test_kerr_cavity_run_rejects_invalid_pulse_parameters(run_kwargs, match):
    cavity = KerrCavity(K=0.5, kappa=0.1, N_cutoff=6)
    with pytest.raises(ValueError, match=match):
        cavity.run(rho_init=qt.ket2dm(qt.fock(6, 0)), **run_kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kappa": -0.1, "N_cutoff": 8}, "kappa must be finite and >= 0"),
        (
            {"kappa": 0.1, "N_cutoff": 8, "loss_time": -1.0},
            "loss_time must be finite and >= 0",
        ),
        ({"kappa": 0.1, "N_cutoff": 0}, "N_cutoff must be a positive integer"),
    ],
)
def test_mzi_rejects_invalid_constructor_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MachZehnderInterferometer(**kwargs)


@pytest.mark.parametrize(
    ("theta_list", "match"),
    [
        (np.array([]), "non-empty"),
        (np.array([0.0, np.nan]), "finite values"),
    ],
)
def test_mzi_scan_rejects_invalid_theta_list(theta_list, match):
    mzi = MachZehnderInterferometer(kappa=0.0, N_cutoff=6)
    with pytest.raises(ValueError, match=match):
        mzi.scan(qt.coherent(6, 1.0), theta_list)


# Cavity and interferometer visual diagnostics


@pytest.mark.visualize
def test_laser_pulse_cavity_plot_demo(assert_no_empty_axes, assert_layout_can_render):
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

    fig = plt.figure(figsize=(6, 4))
    plt.plot(tlist, photon_numbers)
    plt.xlabel("time")
    plt.ylabel("<n>")
    plt.title("Driven Kerr cavity: photon number vs time")
    assert_no_empty_axes(fig)
    assert_layout_can_render(fig)
    plt.show()


@pytest.mark.visualize
def test_full_cavity_multipanel_plot_demo(assert_no_empty_axes, assert_layout_can_render):
    N_cutoff = 12
    alpha = 1.5
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.linspace(0, 2 * np.pi, 60)
    results = MachZehnderInterferometer(kappa=0.2, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(theta_list, results["n1"])
    axes[0].set_title("Mean photon number, arm 1")
    axes[1].plot(theta_list, results["n2"])
    axes[1].set_title("Mean photon number, arm 2")
    axes[2].plot(theta_list, results["parity1"])
    axes[2].set_title("Parity, arm 1")
    plt.tight_layout()
    assert_no_empty_axes(fig)
    assert_layout_can_render(fig)
    plt.show()


# Cavity and interferometer simulations


def test_triggered_cavity_end_to_end():
    cv_circuit = Circuit().add_mode("c")
    cv_circuit.add_gate(
        Gate(
            name="Squeezer",
            transform=squeeze,
            modes=("c",),
            kwargs={"r": 0.1, "theta": 0.0},
        )
    )
    initial_state = cv_circuit.run(GaussianState.vacuum(("c",)))

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

    rho_final_non_gaussian = FockGates.photon_subtraction(rho_kerr_cat, N_cutoff=N_fock)
    purity = (rho_final_non_gaussian * rho_final_non_gaussian).tr().real
    assert 0.0 < purity <= 1.0 + 1e-9


def test_decoherence_mzi_parity_visibility_drops_with_loss():
    start_time = perf_counter()
    N_cutoff = 9
    alpha = 1.5
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    theta_list = np.linspace(0, 2 * np.pi, 50)
    results_clean = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )
    results_noisy = MachZehnderInterferometer(kappa=0.4, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )

    tail = slice(len(theta_list) // 2, None)
    visibility_clean = np.ptp(np.array(results_clean["parity1"])[tail])
    visibility_noisy = np.ptp(np.array(results_noisy["parity1"])[tail])
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

    assert not np.allclose(result["n1"][0], result["n1"][1], atol=1e-8)


def test_mzi_scan_accepts_a_density_matrix_input_matching_the_ket_result():
    N_cutoff = 10
    alpha = 1.3
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.array([0.0, 0.6, 1.9, 3.1])
    mzi = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff, loss_time=0.0)

    result_ket = mzi.scan(psi_cat, theta_list)
    result_dm = mzi.scan(qt.ket2dm(psi_cat), theta_list)

    for key in ("n1", "n2", "parity1"):
        np.testing.assert_allclose(result_dm[key], result_ket[key], atol=1e-10)


def test_lossy_kerr_cat_feeds_directly_into_mzi_scan():
    N_cutoff = 12
    tlist = np.linspace(0, 4, 60)
    states = KerrCavity(K=0.4, kappa=0.08, N_cutoff=N_cutoff).run(
        rho_init=qt.ket2dm(qt.fock(N_cutoff, 0)),
        tlist=tlist,
        amp=4.0,
        t0=1.5,
        sigma=0.6,
    )
    rho_decohered = states[-1]
    assert not rho_decohered.isket
    assert rho_decohered.tr() == pytest.approx(1.0, abs=1e-6)
    purity = (rho_decohered * rho_decohered).tr().real
    assert purity < 1.0 - 1e-6

    theta_list = np.linspace(0, 2 * np.pi, 40)
    result = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff, loss_time=0.0).scan(
        rho_decohered, theta_list
    )
    assert len(result["n1"]) == len(theta_list)
    assert all(np.isfinite(result["n1"]))
    assert all(np.isfinite(result["parity1"]))


@pytest.mark.visualize
def test_kerr_cavity_decoherence_through_mzi_fringe_visibility_demo():
    N_cutoff = 16
    tlist = np.linspace(0, 4, 80)
    theta_list = np.linspace(0, 2 * np.pi, 100)

    fig, (ax_purity, ax_fringes) = plt.subplots(1, 2, figsize=(12, 5))
    purities = []
    labels = ["lossless\n(kappa=0)", "mildly lossy\n(kappa=0.1)", "lossy\n(kappa=0.2)"]
    for kappa_cav, label, color in zip(
        [0.0, 0.1, 0.2], labels, ["darkgreen", "darkorange", "crimson"], strict=True
    ):
        states = KerrCavity(K=0.5, kappa=kappa_cav, N_cutoff=N_cutoff).run(
            rho_init=qt.ket2dm(qt.fock(N_cutoff, 0)),
            tlist=tlist,
            amp=5.0,
            t0=2.0,
            sigma=0.8,
        )
        rho_cat = states[-1]
        purity = (rho_cat * rho_cat).tr().real
        purities.append(purity)

        result = MachZehnderInterferometer(
            kappa=0.0, N_cutoff=N_cutoff, loss_time=0.0
        ).scan(rho_cat, theta_list)
        ax_fringes.plot(
            theta_list / np.pi, result["parity1"], label=label, color=color, lw=2
        )

    ax_purity.bar(labels, purities, color=["darkgreen", "darkorange", "crimson"])
    ax_purity.set_ylabel("Cavity output purity Tr(ρ²)")
    ax_purity.set_title("Cat-state purity after the lossy cavity")
    ax_purity.set_ylim(0, 1.05)

    ax_fringes.axhline(0, color="black", lw=0.5)
    ax_fringes.set_xlabel(r"MZI phase $\theta$ ($\times \pi$)")
    ax_fringes.set_ylabel("Parity expectation value")
    ax_fringes.set_title("Fringe visibility vs. cavity loss")
    ax_fringes.legend()
    ax_fringes.grid(True, ls="--")

    fig.suptitle("Lossy Kerr cavity -> Mach-Zehnder: decoherence eats the fringes")
    plt.tight_layout()
    plt.show()

    assert purities[0] > purities[1] > purities[2]


# Kerr and cat-state simulations


@pytest.mark.visualize
def test_kerr_cat_state_generation(plot_enabled):
    N_cutoff = 35
    rho_vacuum = qt.ket2dm(qt.fock(N_cutoff, 0))
    tlist = np.linspace(0, 6, 200)

    states = KerrCavity(K=0.5, kappa=0.01, N_cutoff=N_cutoff).run(
        rho_init=rho_vacuum,
        tlist=tlist,
        amp=5.0,
        t0=2.0,
        sigma=0.8,
    )
    assert len(states) == len(tlist)
    assert states[-1].tr() == pytest.approx(1.0, abs=1e-6)

    if plot_enabled:
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


@pytest.mark.visualize
def test_cat_state_single_shot_through_mzi():
    N_cutoff = 22
    alpha = 2
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()

    a1 = qt.tensor(qt.destroy(N_cutoff), qt.qeye(N_cutoff))
    a2 = qt.tensor(qt.qeye(N_cutoff), qt.destroy(N_cutoff))

    H_BS = (1j * np.pi / 4) * (a1.dag() * a2 + a1 * a2.dag())
    U_BS = H_BS.expm()

    psi_in = qt.tensor(psi_cat, qt.fock(N_cutoff, 0))
    psi_after_bs1 = U_BS * psi_in

    theta = np.pi / 4
    U_phase = (1j * theta * a1.dag() * a1).expm()
    psi_after_phase = U_phase * psi_after_bs1
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


@pytest.mark.visualize
def test_cat_mzi_phase_scan_fringes(assert_no_empty_axes, assert_layout_can_render):
    N_cutoff = 22
    alpha = 4.0 + 2j
    psi_cat = (qt.coherent(N_cutoff, alpha) + qt.coherent(N_cutoff, -alpha)).unit()
    theta_list = np.linspace(0, 2 * np.pi, 200)

    results = MachZehnderInterferometer(kappa=0.0, N_cutoff=N_cutoff).scan(
        psi_cat, theta_list
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

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
    assert_no_empty_axes(fig)
    assert_layout_can_render(fig)
    plt.show()

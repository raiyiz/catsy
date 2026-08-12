import numpy as np
import pytest

from .composition import OpticalComponent, OpticalSetup
from .states import GaussianOperations, compute_duan_inseparability

# ---------------------------------------------------------------------------
# Layout assembly + serialization
# ---------------------------------------------------------------------------


def test_beam_splitter_registers_both_ports():
    setup = OpticalSetup("Bench").beam_splitter("BS1", port_a="a", port_b="b", eta=0.5)
    assert setup.registered_ports == {"a", "b"}
    assert setup.components[0].op_type == "BeamSplitter"
    assert setup.components[0].kwargs == {"eta": 0.5}


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
    comp = OpticalComponent("BS1", "BeamSplitter", ("a", "b"), {"eta": 0.5})
    assert OpticalComponent.from_dict(comp.to_dict()) == comp


def test_component_rejects_wrong_port_count():
    with pytest.raises(ValueError, match="exactly 2 port"):
        OpticalComponent("BS1", "BeamSplitter", ("a",), {"eta": 0.5})

    with pytest.raises(ValueError, match="exactly 1 port"):
        OpticalComponent("Loss", "Loss", ("a", "b"), {"eta": 0.9})


def test_component_rejects_duplicate_ports():
    with pytest.raises(ValueError, match="same port"):
        OpticalComponent("BS1", "BeamSplitter", ("a", "a"), {"eta": 0.5})


def test_component_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="eta"):
        OpticalComponent("BS1", "BeamSplitter", ("a", "b"), {"eta": 1.1})

    with pytest.raises(ValueError, match="finite"):
        OpticalComponent(
            "Phase", "PhaseRotation", ("a",), {"phi": np.inf}
        )


def test_component_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown optical component"):
        OpticalComponent("Mystery", "Unknown", ("a",), {})


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_process_beam_rejects_empty_setup():
    setup = OpticalSetup("Empty Bench")
    vacuum = GaussianOperations.create_vacuum(modes=("a",))
    with pytest.raises(ValueError):
        setup.process_beam(vacuum)


def test_mzi_setup_preserves_purity_for_coherent_input():
    mzi = OpticalSetup("MZI Node")
    mzi.beam_splitter("BS1", port_a="line_1", port_b="line_2", eta=0.5)
    mzi.phase_shifter("Phase", port="line_2", phi=0.785)
    mzi.beam_splitter("BS2", port_a="line_1", port_b="line_2", eta=0.5)

    coherent_in = GaussianOperations.create_coherent(
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

    epr_in = GaussianOperations.create_epr_pair(mode_a="line_1", mode_b="line_2", r=1.2)
    duan_before = compute_duan_inseparability(epr_in, "line_1", "line_2")

    result = setup.process_beam(epr_in)
    duan_after = compute_duan_inseparability(result, "line_1", "line_2")

    # Loss always moves the witness toward the separability bound...
    assert duan_after > duan_before
    # ...but 10% loss on one arm isn't enough to destroy r=1.2 entanglement.
    assert duan_after < 2.0


def test_process_beam_rejects_input_state_missing_a_registered_port():
    setup = OpticalSetup("Bench").beam_splitter("BS1", port_a="a", port_b="b", eta=0.5)
    mismatched = GaussianOperations.create_vacuum(modes=("a",))
    with pytest.raises(ValueError):
        setup.process_beam(mismatched)


# ---------------------------------------------------------------------------
# Schematic rendering
# ---------------------------------------------------------------------------


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
    assert "BS" in schematic
    assert "LOSS" in schematic
    assert "PHASE" in schematic
    assert "line_1" in schematic and "line_2" in schematic


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

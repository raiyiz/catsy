import pytest

from catsy.modes import Mode, ModeNamespace


def test_standalone_mode_has_no_owner_or_index():
    mode = Mode("my-mode")
    assert mode.name == "my-mode"
    assert mode.index is None
    assert mode.owner is None
    assert repr(mode) == "Mode('my-mode')"


def test_mode_is_immutable():
    mode = Mode("a")
    with pytest.raises((AttributeError, TypeError)):
        mode.name = "b"


def test_owned_mode_identity_does_not_use_value_equality():
    owner = object()
    first = Mode("a", index=0, owner=owner)
    second = Mode("a", index=0, owner=owner)
    assert first is not second
    assert first != second


def test_mode_namespace_is_named_and_ordered():
    owner = object()
    signal = Mode("signal", index=0, owner=owner)
    idler = Mode("idler", index=1, owner=owner)
    modes = ModeNamespace((signal, idler))

    assert modes.signal is signal
    assert modes.idler is idler
    assert modes[0] is signal
    assert modes["idler"] is idler
    assert tuple(modes) == (signal, idler)
    assert len(modes) == 2
    assert "signal" in modes
    assert idler in modes

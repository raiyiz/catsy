import pytest

from catsy import Mode
from catsy.optics import Mode as OpticsMode


def test_mode_is_defined_in_optics_and_publicly_exported():
    assert Mode is OpticsMode


def test_mode_identity_equality():
    first = Mode("a")
    second = Mode("a")

    assert first.name == second.name
    assert first is not second
    assert first != second


def test_mode_owner_defaults_to_none():
    mode = Mode("a")

    assert mode.owner is None


@pytest.mark.parametrize("name", ["", " ", "\t", "\n"])
def test_mode_rejects_blank_names(name: str):
    with pytest.raises(ValueError, match="non-empty string"):
        Mode(name)

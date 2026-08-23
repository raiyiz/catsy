import pytest

from catsy import Mode
from catsy.optics import Mode as OpticsMode


def test_mode_is_defined_in_optics_and_publicly_exported():
    assert Mode is OpticsMode


def test_mode_identity_equality_and_hashing():
    first = Mode("a")
    second = Mode("a")

    assert first.name == second.name
    assert first is not second
    assert first != second
    assert hash(first) != hash(second)
    assert len({first, second}) == 2


def test_mode_owner_defaults_to_none_and_is_assignable():
    mode = Mode("a")

    assert mode.owner is None
    owner = object()
    mode.owner = owner  # type: ignore[assignment]
    assert mode.owner is owner


@pytest.mark.parametrize("name", ["", " ", "\t", "\n", "\r\n"])
def test_mode_rejects_blank_names(name: str):
    with pytest.raises(ValueError, match="non-empty string"):
        Mode(name)


@pytest.mark.parametrize("name", [None, 1, object(), b"a"])
def test_mode_rejects_non_string_names(name: object):
    with pytest.raises(ValueError, match="non-empty string"):
        Mode(name)  # type: ignore[arg-type]


def test_mode_preserves_non_blank_name_verbatim():
    name = "  signal arm  "
    mode = Mode(name)

    assert mode.name == name


def test_mode_is_slot_based():
    mode = Mode("a")

    with pytest.raises(AttributeError):
        mode.unexpected_attribute = True  # type: ignore[attr-defined]

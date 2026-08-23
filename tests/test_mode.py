from catsy import Mode
from catsy.modes import Mode as RuntimeMode


def test_mode_is_the_single_public_runtime_type():
    assert Mode is RuntimeMode


def test_mode_identity_equality():
    first = Mode("a")
    second = Mode("a")

    assert first.name == second.name
    assert first is not second
    assert first != second
    assert hash(first) != hash(second)


def test_mode_rejects_empty_names():
    for name in ("", " ", "\t", "\n"):
        try:
            Mode(name)
        except ValueError:
            pass
        else:
            raise AssertionError("Mode should reject a blank name")


def test_mode_can_be_adopted_once():
    class Circuit:
        pass

    circuit = Circuit()
    mode = Mode("a")

    mode.adopt(circuit)
    assert mode.owner is circuit

    mode.adopt(circuit)
    assert mode.owner is circuit


def test_mode_cannot_be_adopted_by_two_circuits():
    class Circuit:
        pass

    first = Circuit()
    second = Circuit()
    mode = Mode("a")
    mode.adopt(first)

    try:
        mode.adopt(second)
    except ValueError:
        pass
    else:
        raise AssertionError("A mode must not have two owners")


def test_mode_release_requires_current_owner():
    class Circuit:
        pass

    first = Circuit()
    second = Circuit()
    mode = Mode("a")
    mode.adopt(first)

    try:
        mode.release(second)
    except ValueError:
        pass
    else:
        raise AssertionError("A different circuit must not release this mode")

    mode.release(first)
    assert mode.owner is None

import pytest

from quam_libs.wiring.instruments import Instruments

visualize = False


def test_opx_plus_and_octave_validation():
    instruments = Instruments()
    instruments.add_opx_plus(cons=1)
    instruments.add_octave(indices=1)


def test_redefinition_validation():
    instruments = Instruments()
    instruments.add_opx_plus(cons=1)
    with pytest.raises(ValueError):
        instruments.add_opx_plus(cons=1)

    instruments = Instruments()
    instruments.add_octave(indices=1)
    with pytest.raises(ValueError):
        instruments.add_octave(indices=1)


def test_slot_filled_validation():
    instruments = Instruments()
    with pytest.raises(ValueError):
        instruments.add_lf_fem(con=1, slots=1)
        instruments.add_lf_fem(con=1, slots=1)

    instruments = Instruments()
    with pytest.raises(ValueError):
        instruments.add_lf_fem(con=1, slots=[1, 2])
        instruments.add_lf_fem(con=1, slots=[2, 3])

    instruments = Instruments()
    instruments.add_lf_fem(con=1, slots=1)
    instruments.add_lf_fem(con=2, slots=1)
    with pytest.raises(ValueError):
        instruments.add_mw_fem(con=1, slots=1)


def test_opx_1000_and_opx_plus_mixing_validation():
    instruments = Instruments()
    instruments.add_lf_fem(con=1, slots=1)
    with pytest.raises(ValueError):
        instruments.add_opx_plus(cons=1)

    instruments = Instruments()
    instruments.add_mw_fem(con=1, slots=1)
    with pytest.raises(ValueError):
        instruments.add_opx_plus(cons=1)

    instruments = Instruments()
    instruments.add_opx_plus(cons=1)
    with pytest.raises(ValueError):
        instruments.add_lf_fem(con=1, slots=1)
    with pytest.raises(ValueError):
        instruments.add_mw_fem(con=1, slots=1)

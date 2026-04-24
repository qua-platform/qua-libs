"""QUA gate-playback helpers for gate set tomography.

The :func:`play_gst_sequence` function implements ``switch/case`` block on 
the preparation, measurement and germ portions of the GST sequence.
It maps integers to a gate (or series of gates) to qubit macros registered 
on the qubit object.

Native gate integer encoding (must match :data:`gst_sequences.PREP_FIDUCIAL_MAP`, 
:data:`gst_sequences.MEAS_FIDUCIAL_MAP` and :data:`gst_sequences.GERM_MAP`)
"""

from __future__ import annotations

from qm.qua import switch_, case_, for_, declare


def play_gst_sequence(qubit, prep_id, meas_id, germ_id, repetition) -> None:
    """Play a GST sequence on *qubit* selected by *prep_id*, *meas_id*, *germ_id* and *repetition*.

    Parameters
    ----------
    qubit : LDQubit
        The qubit object with registered gate macros on its ``xy`` channel.
    """
    germ_rep = declare(int)

    # switch on preparation fiducial
    with switch_(prep_id, unsafe=True):
        with case_(0):
            pass
        with case_(1):
            qubit.x90()
        with case_(2):
            qubit.y90()
        with case_(3):
            qubit.x90()
            qubit.x90()
        with case_(4):
            qubit.x90()
            qubit.x90()
            qubit.x90()
        with case_(5):
            qubit.y90()
            qubit.y90()
            qubit.y90()
    
    # repetition and switch on germ fiducial
    with for_(germ_rep, 0, germ_rep < repetition, germ_rep + 1):
        with switch_(germ_id, unsafe=True):
            with case_(0):
                pass
            with case_(1):
                qubit.x90()
            with case_(2):
                qubit.y90()
            with case_(3):
                qubit.x90()
                qubit.y90()
            with case_(4):
                qubit.x90()
                qubit.x90()
                qubit.y90()
            with case_(5):
                # TODO: add case for identity gate
                pass
    
    # switch on measurement fiducial
    with switch_(meas_id, unsafe=True):
        with case_(0):
            pass
        with case_(1):
            qubit.x90()
        with case_(2):
            qubit.y90()
        with case_(3):
            qubit.x90()
            qubit.x90()
        with case_(4):
            qubit.x90()
            qubit.x90()
            qubit.x90()
        with case_(5):
            qubit.y90()
            qubit.y90()
            qubit.y90()
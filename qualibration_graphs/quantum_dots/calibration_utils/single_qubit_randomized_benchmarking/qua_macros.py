"""QUA gate-playback helpers for single-qubit randomized benchmarking.

The :func:`play_rb_gate` function implements a 12-entry ``switch/case``
that maps gate integers (produced by the Clifford decomposition tables)
to physical pulse operations or virtual frame rotations on the qubit's
XY channel.

Native gate integer encoding (must match :data:`clifford_tables.NATIVE_GATE_MAP`):

    0: x90     1: x180    2: -x90    3: -x180
    4: y90     5: y180    6: -y90    7: -y180
    8: z90     9: z180   10: z270   11: idle
"""

from __future__ import annotations

from qm.qua import switch_, case_, frame_rotation_2pi


def play_rb_gate(qubit, gate_int) -> None:
    """Play a single native gate on *qubit* selected by *gate_int*.

    This generates a QUA ``switch_/case_`` block that dispatches to the
    correct pulse operation or frame rotation.  Physical gates (cases 0–7)
    play calibrated Gaussian pulses on ``qubit.xy``.  Virtual Z gates
    (cases 8–10) apply frame rotations with zero duration.  The idle gate
    (case 11) waits one clock cycle (4 ns).

    Parameters
    ----------
    qubit : LDQubit
        The qubit object whose ``xy`` channel will be driven.
    gate_int : QUA int variable
        Gate index (0–11) from the Clifford decomposition tables.
    """
    xy = qubit.xy

    with switch_(gate_int, unsafe=True):
        with case_(0):
            xy.play("x90")
        with case_(1):
            xy.play("x180")
        with case_(2):
            xy.play("-x90")
        with case_(3):
            xy.play("-x180")
        with case_(4):
            xy.play("y90")
        with case_(5):
            xy.play("y180")
        with case_(6):
            xy.play("-y90")
        with case_(7):
            xy.play("-y180")
        with case_(8):
            frame_rotation_2pi(0.25, xy.name)
        with case_(9):
            frame_rotation_2pi(0.5, xy.name)
        with case_(10):
            frame_rotation_2pi(0.75, xy.name)
        with case_(11):
            xy.wait(4)

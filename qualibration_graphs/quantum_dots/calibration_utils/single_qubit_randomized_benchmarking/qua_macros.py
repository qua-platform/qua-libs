"""QUA gate-playback helpers for single-qubit randomized benchmarking.

The :func:`play_rb_gate` function implements a 9-entry ``switch/case``
that maps gate integers (produced by the Clifford decomposition tables)
to qubit macros registered on the qubit object.

The macros themselves enforce the single-calibration-source principle:

  - X rotations: ``amplitude_scale = theta / 180`` on the calibrated X180 pulse
  - Y rotations: frame shift +90°, play X equivalent, undo frame shift
  - Z rotations: pure frame rotation (zero hardware duration)

Native gate integer encoding (must match :data:`clifford_tables.NATIVE_GATE_MAP`):

    0: x90      1: x180    2: x_neg90
    3: y90      4: y180    5: y_neg90
    6: z90      7: z180    8: z_neg90
"""

from __future__ import annotations

from qm.qua import switch_, case_


def play_rb_gate(qubit, gate_int) -> None:
    """Play a single native gate on *qubit* selected by *gate_int*.

    Dispatches to the qubit's registered macros using canonical names
    (``x90``, ``x180``, ``x_neg90``, ``y90``, ``y180``, ``y_neg90``,
    ``z90``, ``z180``, ``z_neg90``).

    Parameters
    ----------
    qubit : LDQubit
        The qubit object with registered gate macros on its ``xy`` channel.
    gate_int : QUA int variable
        Gate index (0–8) from the Clifford decomposition tables.
    """
    with switch_(gate_int, unsafe=True):
        with case_(0):
            qubit.x90()
        with case_(1):
            qubit.x180()
        with case_(2):
            qubit.x_neg90()
        with case_(3):
            qubit.y90()
        with case_(4):
            qubit.y180()
        with case_(5):
            qubit.y_neg90()
        with case_(6):
            qubit.z90()
        with case_(7):
            qubit.z180()
        with case_(8):
            qubit.z_neg90()

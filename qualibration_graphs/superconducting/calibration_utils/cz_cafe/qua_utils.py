"""QUA helpers for playing compiled CAFE circuits.

Every frame rotation is a compile-time constant, selected with ``switch_`` like the gates of
the two-qubit RB node. No angle is loaded from a QUA array or held in a QUA variable while
the pulses play.
"""

import numpy as np
from qm.qua import case_, switch_

# Frame rotations smaller than this (in units of 2π) are skipped
_MIN_ANGLE_2PI = 1e-9


def play_layer(qubit, angles_2pi) -> None:
    """Play ``Rz(a) · Ry(π/2) · Rz(b) · Ry(π/2) · Rz(c)`` on one qubit.

    ``angles_2pi`` holds (a, b, c) in units of 2π. Z rotations are virtual frame rotations.
    """
    a, b, c = (float(angle) for angle in angles_2pi)
    # Time order: Rz(c), y90, Rz(b), y90, Rz(a)
    for angle, pulse in ((c, "y90"), (b, "y90"), (a, None)):
        if abs(angle) > _MIN_ANGLE_2PI:
            qubit.xy.frame_rotation_2pi(angle)
        if pulse is not None:
            qubit.xy.play(pulse)


def play_layer_cases(qubit_pair, case_var, layers: np.ndarray) -> None:
    """Play on both qubits of the pair the layer selected by the QUA int ``case_var``.

    ``layers`` has shape (n_cases, 2 qubits, 3), qubit 0 being ``qubit_control``. ``case_var``
    must lie in [0, n_cases).
    """
    with switch_(case_var, unsafe=True):
        for i, layer in enumerate(layers):
            with case_(i):
                play_layer(qubit_pair.qubit_control, layer[0])
                play_layer(qubit_pair.qubit_target, layer[1])
    qubit_pair.align()

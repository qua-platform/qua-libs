"""QUA gate-playback macros for XEB experiments.

Maps gate integers (0, 1, 2) to spin qubit macros via QUA switch/case.
The gate_set parameter is evaluated at QUA program generation time
(Python if/else), not at QUA runtime.

Gate integer encoding (must match gateset.py):
    0: SX  → qubit.x90()
    1: SY  → qubit.y90()
    2: SW  → qubit.z_neg90() then qubit.x90()   (gate_set="sw")
       T   → qubit.xy.frame_rotation_2pi(0.125)  (gate_set="t")
"""

from __future__ import annotations

from typing import Literal

from qm.qua import switch_, case_


def play_xeb_gate(
    qubit, gate_int, gate_set: Literal["sw", "t"] = "sw"
) -> None:
    """Play a single XEB gate on *qubit* selected by *gate_int*.

    Parameters
    ----------
    qubit : LDQubit
        The qubit object with registered gate macros.
    gate_int : QUA int variable
        Gate index (0-2) from random generation.
    gate_set : {"sw", "t"}
        Which gate set is active. Evaluated at program generation time.
    """
    with switch_(gate_int, unsafe=True):
        with case_(0):
            qubit.x90()
        with case_(1):
            qubit.y90()
        with case_(2):
            if gate_set == "sw":
                qubit.z_neg90()
                qubit.x90()
            else:
                qubit.xy.frame_rotation_2pi(0.125)

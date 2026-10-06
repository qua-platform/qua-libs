"""QUA-side helpers for the joint readout duration x power optimization.

``readout_config_override`` lengthens the readout pulse for the duration of config
generation. The node exists to *choose* an integration duration, so the sweep has to be
able to reach past the length currently in the state -- otherwise a 500 ns readout can
never discover that 1500 ns is better. Lengthening the pulse also removes the unequal
readout length problem for free, since the qubits in the state currently run anywhere
between 200 ns and 1200 ns and accumulated demodulation needs one shared maximum.

``save_accumulated_quadratures`` recombines what ``resonator.measure_accumulated`` leaves in
its four arrays and streams one I/Q pair per integration duration.
"""

from contextlib import contextmanager
from typing import Sequence

from qm.qua import assign, declare, fixed, for_, save

DEFAULT_INTEGRATION_WEIGHTS = "#./default_integration_weights"


def raw_integration_weights(pulse):
    """The stored ``integration_weights``, without resolving a quam reference.

    Reading ``pulse.integration_weights`` resolves ``'#./default_integration_weights'`` into
    the materialised list it points at, so it can never be compared against the reference
    itself. Detecting custom weights -- anything other than the flat defaults -- needs the raw value.
    """
    return pulse.get_unreferenced_value("integration_weights")


def has_custom_integration_weights(pulse) -> bool:
    """Whether this pulse carries weights written by a node rather than the default reference."""
    return raw_integration_weights(pulse) != DEFAULT_INTEGRATION_WEIGHTS


def set_integration_weights(pulse, value) -> None:
    """Assign ``integration_weights``, clearing any existing reference first.

    quam refuses to overwrite a reference in place; it has to be set to None before it can
    take a new value, whether that value is another reference or a literal list.
    """
    pulse.integration_weights = None
    pulse.integration_weights = value


@contextmanager
def readout_config_override(qubits: Sequence, operation: str, length_in_ns: int, log_callable=None):
    """Temporarily set every selected qubit's readout pulse to the sweep's maximum length.

    Custom integration weights are reset to the defaults at the same time: accumulated
    demodulation at fine chunks requires flat weights, and previously optimized weights span
    the *old* pulse length, so they no longer tile the lengthened pulse.

    The originals are always restored, so this mutation cannot escape config generation. It
    is deliberately not recorded as a state update -- the node's own state update decides
    separately, and explicitly, what is written back.
    """
    saved = []
    for qubit in qubits:
        pulse = qubit.resonator.operations[operation]
        saved.append((qubit, pulse, pulse.length, raw_integration_weights(pulse)))
    try:
        for qubit, pulse, old_length, old_weights in saved:
            pulse.length = int(length_in_ns)
            if old_weights != DEFAULT_INTEGRATION_WEIGHTS:
                if log_callable is not None:
                    log_callable(
                        f"{qubit.name}: resetting custom integration weights to the defaults for the "
                        f"sweep (they span the previous {old_length} ns pulse, not {length_in_ns} ns)."
                    )
                set_integration_weights(pulse, DEFAULT_INTEGRATION_WEIGHTS)
        yield
    finally:
        for qubit, pulse, old_length, old_weights in saved:
            pulse.length = old_length
            if raw_integration_weights(pulse) != old_weights:
                set_integration_weights(pulse, old_weights)


def declare_recombination_variables():
    """The two scratch variables ``save_accumulated_quadratures`` recombines through."""
    return declare(fixed), declare(fixed)


def save_accumulated_quadratures(path_arrays, num_durations: int, index, scratch, I_stream, Q_stream) -> None:
    """Recombine the four accumulated components into I/Q and stream one value per duration.

    ``I = II + IQ`` and ``Q = QI + QQ`` is the pairing a full complex demodulation performs
    internally, so element ``k`` is exactly what a conventional measurement of length
    ``(k + 1) * chunk`` would have returned.
    """
    II, IQ, QI, QQ = path_arrays
    I_value, Q_value = scratch
    with for_(index, 0, index < num_durations, index + 1):
        assign(I_value, II[index] + IQ[index])
        assign(Q_value, QI[index] + QQ[index])
        save(I_value, I_stream)
        save(Q_value, Q_stream)

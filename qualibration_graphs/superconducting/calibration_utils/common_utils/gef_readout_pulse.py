"""Shared GEF-readout helpers: pulse setup, resonator IF, and reset.

Three-level discrimination needs a longer integration than the g/e readout, so these nodes
play a dedicated ``readout_GEF`` pulse instead of ``readout``. When the state does not carry
one yet it is derived from ``readout`` here.

A state built by ``quam_builder`` already contains a ``readout_GEF`` seeded with the builder
defaults (``length=2000``, ``amplitude=0.01``), which survives populate scripts that only tune
``readout``. That leftover drives the resonator far below the calibrated readout power, so the
g/e/f blobs never separate and the fits fail for reasons that are invisible in the node output.
:func:`ensure_gef_readout_pulse` therefore also reports how an existing pulse compares to
``readout``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

# quam_builder's untouched default readout_GEF, reported whatever the readout amplitude is.
_BUILDER_DEFAULT_LENGTH = 2000
_BUILDER_DEFAULT_AMPLITUDE = 0.01
# Below this ratio of readout_GEF to readout amplitude the pulse is reported as suspect.
_AMPLITUDE_RATIO_WARNING = 0.25


def ensure_gef_readout_pulse(qubits, log_callable: Optional[Callable] = None) -> None:
    """Give every qubit a ``readout_GEF`` operation and report suspect existing ones.

    Missing operations are derived from ``readout``: 1.5x the length (rounded to a multiple of
    4 ns) and the same amplitude, with the g/e thresholds cleared since they do not apply to
    three-level discrimination.

    Args:
        qubits: The qubits the node operates on.
        log_callable: Where to report a suspect pulse, typically ``node.log``. It is called as
            ``log_callable(message, level="warning")``, so it must accept that keyword. Nothing
            is reported if None.
    """
    for qubit in qubits:
        readout_op = qubit.resonator.operations["readout"]
        gef_op = qubit.resonator.operations.get("readout_GEF")

        if gef_op is None:
            new_length = int(round(readout_op.length * 1.5 / 4) * 4)  # multiple of 4 ns
            qubit.resonator.operations["readout_GEF"] = dataclasses.replace(
                readout_op,
                length=new_length,
                threshold=None,
                rus_exit_threshold=None,
            )
            continue

        if log_callable is None:
            continue

        reason = _suspect_pulse_reason(gef_op, readout_op)
        if reason is not None:
            log_callable(
                f"{qubit.name}: {reason} Seed readout_GEF from the calibrated readout before "
                f"running this node, otherwise the g/e/f blobs are unlikely to separate.",
                level="warning",
            )


def _suspect_pulse_reason(gef_op, readout_op) -> Optional[str]:
    """Describe why ``gef_op`` looks unusable for g/e/f readout, or None if it looks fine.

    Amplitudes are read defensively: a pulse shape without an amplitude field, or one whose
    amplitude is unset, is left alone rather than raising inside a node's program build.
    """
    gef_amplitude = getattr(gef_op, "amplitude", None)
    readout_amplitude = getattr(readout_op, "amplitude", None)

    if gef_amplitude is None:
        return None

    if gef_op.length == _BUILDER_DEFAULT_LENGTH and gef_amplitude == _BUILDER_DEFAULT_AMPLITUDE:
        return (
            f"readout_GEF is still the untouched quam_builder default "
            f"(length {_BUILDER_DEFAULT_LENGTH} ns, amplitude {_BUILDER_DEFAULT_AMPLITUDE})."
        )

    if not readout_amplitude:
        return None

    ratio = abs(gef_amplitude) / abs(readout_amplitude)
    if ratio < _AMPLITUDE_RATIO_WARNING:
        return (
            f"readout_GEF amplitude {gef_amplitude:.4g} is far below the readout amplitude "
            f"{readout_amplitude:.4g} (ratio {ratio:.3g})."
        )

    return None


def gef_readout_frequency(qubit, extra_detuning: float = 0.0) -> float:
    """Resonator IF for a GEF readout: intermediate frequency plus ``GEF_frequency_shift``.

    ``extra_detuning`` is for a frequency sweep around the current GEF operating point
    (node 14). A missing shift is treated as 0.
    """
    shift = qubit.resonator.GEF_frequency_shift
    if shift is None:
        shift = 0
    return qubit.resonator.intermediate_frequency + shift + extra_detuning


def set_gef_readout_frequency(qubit, extra_detuning: float = 0.0) -> None:
    """Point the resonator NCO at the GEF readout frequency for this shot."""
    qubit.resonator.update_frequency(gef_readout_frequency(qubit, extra_detuning))


def reset_for_gef(qubit, reset_type: str, simulate: bool, u: Any) -> None:
    """Thermalize, or run ``active_gef`` and restore the GEF-detuned resonator IF.

    ``reset_qubit_active_gef``'s internal ``readout_state_gef`` leaves the resonator IF
    unshifted on return. GEF nodes measure off the g/e readout frequency, so the shift
    has to be put back before the next ``measure``.
    """
    if reset_type == "thermal":
        qubit.wait(2 * qubit.thermalization_time * u.ns)  # longer wait for |f> thermalization
        return
    qubit.reset(reset_type, simulate)
    set_gef_readout_frequency(qubit)

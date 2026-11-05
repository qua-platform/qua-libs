import logging
from typing import Optional

import numpy as np
from quam_builder.architecture.superconducting.qubit import AnyTransmon

__all__ = ["get_pulse_area", "get_qubit_probe_amplitude_factor"]


def get_pulse_area(pulse) -> float:
    """
    Returns the area of a QUAM pulse, in units of amplitude times samples.

    Square pulses expose a scalar waveform, shaped pulses (e.g. DragCosine) a sampled array, so both cases are
    handled here. Only the in-phase component is taken into account, since the out-of-phase (DRAG) component does
    not contribute to the rotation angle.

    Parameters
    ----------
    pulse
        The QUAM pulse whose area should be computed.

    Returns
    -------
    float
        The pulse area in units of amplitude times samples.
    """
    waveform = np.atleast_1d(np.asarray(pulse.waveform_function()))
    if waveform.size == 1:
        # Square pulses return a single sample that is held for the whole pulse length
        return float(np.real(waveform[0])) * pulse.length
    return float(np.sum(np.real(waveform)))


def get_qubit_probe_amplitude_factor(
    qubit: AnyTransmon,
    probe_operation: str,
    probe_length_in_ns: int,
    pi_fraction: float,
    pi_pulse_operation: str = "x180",
    log_callable: Optional[callable] = None,
) -> float:
    """
    Returns the amplitude pre-factor giving the qubit probe pulse a pulse area of `pi_fraction` times a pi pulse.

    The probe pulse is played with an overridden duration, so its area is rescaled accordingly before being
    compared with the area of the calibrated pi pulse.

    Parameters
    ----------
    qubit : AnyTransmon
        The qubit whose xy drive plays the probe pulse.
    probe_operation : str
        The name of the (square) xy operation used to probe the qubit.
    probe_length_in_ns : int
        The duration the probe pulse is played for, in ns.
    pi_fraction : float
        The targeted pulse area in units of the area of a pi pulse.
    pi_pulse_operation : str
        The name of the calibrated pi pulse operation. Default is "x180".
    log_callable : callable, optional
        Logger used to warn when the resulting pre-factor is out of the allowed range. If None, a default logger
        is used.

    Returns
    -------
    float
        The amplitude pre-factor to pass to `qubit.xy.play(probe_operation, amplitude_scale=...)`.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).warning

    pi_pulse = qubit.xy.operations[pi_pulse_operation]
    probe_pulse = qubit.xy.operations[probe_operation]
    # Mean amplitude of the probe pulse, so that its area at the overridden duration can be derived
    probe_mean_amplitude = get_pulse_area(probe_pulse) / probe_pulse.length
    probe_area = probe_mean_amplitude * probe_length_in_ns
    if probe_area == 0:
        raise ValueError(f"The operation '{probe_operation}' of {qubit.xy.name} has a null pulse area.")

    amplitude_factor = pi_fraction * get_pulse_area(pi_pulse) / probe_area
    if not -2 <= amplitude_factor < 2:
        log_callable(
            f"The qubit probe amplitude factor derived for {qubit.name} ({amplitude_factor:.3f}) is outside the "
            f"allowed [-2; 2) range. Either lengthen 'qubit_probe_len_in_ns' or set "
            f"'qubit_probe_amplitude_factor' explicitly."
        )
    return amplitude_factor

"""Parameters for the f-LRU (|f0> -> |g1>) microwave pump calibration."""

from typing import List, Optional

import numpy as np
from pydantic import model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Sweep of the LRU pump: duration x amplitude x frequency, measured with g/e/f readout."""

    num_shots: int = 100
    """Number of averages. Default is 100."""
    operation: str = "lru"
    """LRU pulse on ``qubit.lru`` (a square pulse; its duration is overridden by the sweep). Default is "lru"."""
    pulse_durations_ns: List[int] = [48, 100, 200]
    """LRU pulse durations to scan, in ns (multiples of 4 ns, >= 16 ns). Default is [48, 100, 200]."""
    min_amp_factor: float = 0.0
    """Minimum LRU amplitude pre-factor (times the ``operation`` amplitude). Default is 0.0."""
    max_amp_factor: float = 1.98
    """Maximum LRU amplitude pre-factor, excluded, < 2. Default is 1.98."""
    amp_factor_step: float = 0.06
    """LRU amplitude pre-factor step. Default is 0.06."""
    frequency_span_in_mhz: float = 200.0
    """Span of the LRU drive frequency sweep, in MHz. Default is 200 MHz."""
    frequency_step_in_mhz: float = 2.0
    """Step of the LRU drive frequency sweep, in MHz. Default is 2 MHz."""
    frequency_detuning_in_mhz: float = 0.0
    """Offset of the frequency sweep centre from ``qubit.lru.RF_frequency``, in MHz. Default is 0."""
    wait_after_lru_ns: Optional[int] = None
    """Wait between the LRU pulse and the readout, letting the photon left in the resonator decay.
    If None, the resonator ``depletion_time``. Default is None."""
    readout_operation: str = "readout_GEF"
    """Resonator pulse used for the g/e/f readout (nodes 14a/14b/15). Default is "readout_GEF"."""
    skip_lru_pulse: bool = False
    """Debug: leave out the LRU pulse and keep the rest of the sequence, including the wait before readout. P(f)
    should then match the |f> population of node 15. Default is False."""
    return_iq_blobs: bool = False
    """Debug: measure readout_operation at the GEF readout frequency, exactly as the classified readout does, but
    save every shot's raw I/Q instead of the g/e/f state. The analysis classifies the shots offline with centers
    refit on this run, fits the resulting populations as usual and adds an IQ-blob figure. Data grows with num_shots x durations x amplitudes x frequencies, so shrink
    the sweep first. Default is False."""
    add_lru_to_state: bool = False
    """Keep the LRU element, and its upconverter on the XY port, in the QUAM state for qubits whose fit succeeds.
    Their calibrated LRU frequency, amplitude and duration are then proposed as state updates. If False, qubits
    without an LRU element get one for this run only and the calibrated values are just logged. Default is False."""
    min_ridge_contrast: float = 0.2
    """Minimum dip in P(f), as a fraction of the prepared P(f), for an amplitude to enter the resonance-line fit.
    Default is 0.2."""
    min_prepared_pf: float = 0.5
    """Minimum P(f) at the lowest LRU amplitude, i.e. how well |f> is prepared and read out. Below it the qubit
    fails without fitting, since there is no |f> population for the LRU to remove. Default is 0.5."""
    min_dip_sigma: float = 5.0
    """Minimum drop of P(f) along the resonance line, in units of the shot noise of the prepared P(f), for a fit
    to succeed. Default is 5."""

    @model_validator(mode="after")
    def _validate_sweep(self):
        """Check durations fit the 4 ns clock and the amplitude pre-factor stays within QUA's [-2, 2)."""
        bad = [d for d in self.pulse_durations_ns if d < 16 or d % 4]
        if bad:
            raise ValueError(f"pulse_durations_ns must be multiples of 4 ns and >= 16 ns, got {bad}.")
        if not -2 <= self.min_amp_factor < self.max_amp_factor <= 2:
            raise ValueError("Amplitude pre-factors must satisfy -2 <= min_amp_factor < max_amp_factor <= 2.")
        return self


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for the f-LRU calibration node."""


def get_amp_factors(node_parameters: NodeSpecificParameters) -> np.ndarray:
    """LRU amplitude pre-factors to sweep (the maximum is excluded, as in the power Rabi nodes)."""
    return np.arange(node_parameters.min_amp_factor, node_parameters.max_amp_factor, node_parameters.amp_factor_step)


def get_detunings(node_parameters: NodeSpecificParameters) -> np.ndarray:
    """LRU drive detunings from ``qubit.lru.RF_frequency`` to sweep, in Hz (integers)."""
    half_span = node_parameters.frequency_span_in_mhz * 1e6 / 2
    step = node_parameters.frequency_step_in_mhz * 1e6
    centre = node_parameters.frequency_detuning_in_mhz * 1e6
    return (centre + np.arange(-half_span, half_span + step / 2, step)).astype(int)

"""Parameters for the flux-activated f-LRU (|f0> -> |e1>) calibration."""

from typing import List, Optional

from pydantic import model_validator
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class FluxNodeSpecificParameters(RunnableParameters):
    """Sweep of the flux tone: duration x amplitude x frequency, measured with g/e/f readout."""

    num_shots: int = 100
    """Number of averages. Default is 100."""
    pulse_durations_ns: List[int] = [500, 1500]
    """Flux-tone durations to scan, in ns (multiples of 4 ns, >= 16 ns). Default is [500, 1500]."""
    flux_amplitude_v: float = 0.2
    """Reference flux-tone amplitude at the port, in volts; the sweep scales it by the pre-factors. The tone plus the
    flux offset must stay within the port's range (+-0.5 V in direct mode, +-2.5 V in amplified mode). Default is 0.2."""
    min_amp_factor: float = 0.0
    """Minimum amplitude pre-factor. Default is 0.0."""
    max_amp_factor: float = 1.98
    """Maximum amplitude pre-factor, excluded, < 2. Default is 1.98."""
    amp_factor_step: float = 0.165
    """Amplitude pre-factor step. Default is 0.165."""
    frequency_span_in_mhz: float = 60.0
    """Span of the swept |f0> -> |e1> drive frequency (twice the flux-tone frequency), in MHz. Default is 60 MHz."""
    frequency_step_in_mhz: float = 0.5
    """Step of the swept |f0> -> |e1> drive frequency, in MHz. Default is 0.5 MHz."""
    frequency_detuning_in_mhz: float = 0.0
    """Offset of the sweep centre from the expected f_r - f_12, in MHz. Default is 0."""
    wait_after_lru_ns: Optional[int] = None
    """Wait between the flux tone and the readout, letting the photon created in |e1> leave the resonator.
    If None, the resonator ``depletion_time``. Default is None."""
    readout_operation: str = "readout_GEF"
    """Resonator pulse used for the g/e/f readout (nodes 14a/14b/15). Default is "readout_GEF"."""
    min_ridge_contrast: float = 0.2
    """Minimum dip in P(f), as a fraction of the prepared P(f), for an amplitude to enter the resonance-line fit.
    Default is 0.2."""
    min_prepared_pf: float = 0.5
    """Minimum P(f) at the lowest amplitude, i.e. how well |f> is prepared and read out. Default is 0.5."""
    min_dip_sigma: float = 5.0
    """Minimum drop of P(f) along the resonance line, in units of the shot noise of the prepared P(f), for a fit
    to succeed. Default is 5."""

    @model_validator(mode="after")
    def _validate_sweep(self):
        """Check durations fit the 4 ns clock and the pre-factors stay within QUA's [-2, 2).

        The port's range depends on its output mode, so node 26b checks it against the QUAM state."""
        bad = [d for d in self.pulse_durations_ns if d < 16 or d % 4]
        if bad:
            raise ValueError(f"pulse_durations_ns must be multiples of 4 ns and >= 16 ns, got {bad}.")
        if not -2 <= self.min_amp_factor < self.max_amp_factor <= 2:
            raise ValueError("Amplitude pre-factors must satisfy -2 <= min_amp_factor < max_amp_factor <= 2.")
        return self


class FluxParameters(
    NodeParameters,
    CommonNodeParameters,
    FluxNodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for the flux-activated f-LRU calibration node."""

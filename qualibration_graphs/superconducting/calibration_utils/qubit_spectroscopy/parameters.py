"""Parameters for qubit spectroscopy calibration (v2)."""

from typing import Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for qubit spectroscopy."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 100
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""
    operation: str = "saturation"
    """Type of operation to perform. Default is "saturation"."""
    operation_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the operation. Default is 1.0."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length."""
    target_peak_width: float = 3e6
    """Target peak FWHM in Hz used to rescale the saturation amplitude. Default 3 MHz."""
    update_pulses_amplitude: bool = False
    """Whether to update the saturation pulse and x180/x90 pulse amplitudes based on the peak width. Default False."""

    # --- Fit quality gates (new in v2) ---
    r2_threshold: float = 0.75
    """Minimum coefficient of determination (R²) inside the wider refit window for the
    fit to count as successful. Default 0.75 — the refit covers ±max(4×FWHM, 10 MHz)
    so a true qubit line plus several FWHM of flat baseline reliably scores > 0.85;
    0.75 gives a safety margin for noise-limited scans. Raise to e.g. 0.9 to be stricter."""
    max_fwhm_mhz: float = 30.0
    """Reject fits whose FWHM exceeds this many MHz (typical qubit lines are 0.1–5 MHz)."""
    min_contrast: float = 0.05
    """Reject fits whose fitted peak / fitted baseline contrast is below this fraction."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for qubit spectroscopy calibration."""

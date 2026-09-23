"""Parameter definitions for the readout quantum efficiency node."""

from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Sweep and analysis parameters for the quantum efficiency measurement."""

    num_shots_ramsey: int = 10000
    """Number of hardware averages for the dephasing (Ramsey) half of each cell. Independent of
    num_shots_snr. Default is 10000."""
    num_shots_snr: int = 10000
    """Number of single shots per prepared state for the SNR half of each cell. These are kept
    individually - the SNR is the ratio of a mean separation to a standard deviation, so a few
    hundred to a few thousand are needed for it to mean anything. Default is 10000."""
    sweep_frequency: bool = False
    """Whether to sweep the measurement-pulse frequency. When False the measurement pulse stays at
    the calibrated readout frequency and only the amplitude axis is swept, which is the fast way to
    get eta at the current setpoint. The detuning axis is then a single point at 0 Hz and
    frequency_span_in_mhz / frequency_step_in_mhz are ignored. Default is False."""
    frequency_span_in_mhz: float = 2.0
    """Span of the readout frequency sweep, centred on the current readout frequency, in MHz. Default is 2.0 MHz."""
    frequency_step_in_mhz: float = 1
    """Step of the readout frequency sweep in MHz. Default is 1 MHz."""
    max_amp_prefactor: float = 0.35
    """Largest measurement-pulse amplitude prefactor, relative to the calibrated readout amplitude. Values above 1
    probe the non-linear regime on purpose; QUA caps the amplitude scale below 2. Default is 0.35."""
    num_amps: int = 11
    """Number of non-zero amplitude prefactors. The eps=0 reference is always prepended, so the amplitude axis has
    num_amps + 1 points. Default is 11."""
    num_phases: int = 31
    """Number of azimuthal angles of the second pi/2 pulse, spread uniformly over a whole number of
    fringes. Default is 31."""
    max_amp_for_fit: Optional[float] = None
    """Largest amplitude prefactor included in the linear-regime fit that yields eta. None auto-detects the linear
    range from the fit residuals. Default is None."""
    linearity_rtol: float = 0.25
    """How far the largest amplitude's SNR/eps and beta/eps^2 may deviate from the median of the
    remaining points before it is dropped as non-linear. Loose on purpose: real sweeps show tens of
    percent of curvature across a decade of amplitude, and a tight tolerance would reject usable
    data and leave the fit to the smallest, noisiest amplitudes. Default is 0.25."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Aggregate node parameters for the readout quantum efficiency node.

    Inherits `multiplexed` from `QubitsExperimentNodeParameters`, but the node forces it to False
    at runtime: a multiplexed weak pulse would dephase every qubit in the batch while the SNR half
    only looks at this qubit's own IQ, so eta would come out low for a reason unrelated to the
    amplifier. Setting `multiplexed = True` here therefore has no effect.
    """

    pass

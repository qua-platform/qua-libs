from qualibrate import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameter class for the resonator linewidth calibration node."""

    num_shots: int = 1000
    """Number of averages to perform. Default is 1000."""

    frequency_span_in_mhz: float = 10.0
    """Span of frequencies to sweep in MHz. Default is 10 MHz."""

    frequency_step_in_mhz: float = 0.05
    """Step size for frequency sweep in MHz. Default is 0.05 MHz."""

    probe_amplitude_scale: float = 0.1
    """Probe amplitude as a scale on each qubit's readout amplitude, in [0, 2). Keep it low so the
    resonator is not saturated. Default is 0.1."""

    probe_skip_ns: int | None = None
    """Start of the probe left out of the integration while the resonator fills up; integrating it biases
    kappa high. None uses 8 / kappa from the stored kappa, or 2000 ns. Multiple of 4."""

    probe_window_ns: int | None = None
    """Integration window after the skip. None uses min(2000 ns, T1 / 10), so |1> barely decays.
    Multiple of 4."""

    measure_excited_state: bool = True
    """Also sweep with the qubit in |1> to get chi and kappa_e. Doubles the run time. Default is True."""

    correct_background_tilt: bool = True
    """Fit and divide out a linear tilt of the off-resonant |S21| before the circle fit. Default is True."""

    fit_cable_delay: bool = True
    """Fit the cable delay; False uses the stored time of flight, which makes Q_c unreliable. Default is True."""

    min_r_squared: float = 0.8
    """Minimum R² of the |S21| fit for success. Default is 0.8."""

    chi_mismatch_warning_fraction: float = 0.2
    """Warn when the fitted chi differs from the stored one (e.g. from node 08a) by more than this. Default is 0.2."""

    kappa_mismatch_warning_fraction: float = 0.2
    """Warn when the fitted kappa differs from the stored one by more than this. Default is 0.2."""

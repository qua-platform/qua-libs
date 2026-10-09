from qualibrate import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameter class for the resonator linewidth calibration node."""

    num_shots: int = 1000
    """Number of averages to perform. The circle fit needs a clean complex trace, so this defaults
    higher than a routine spectroscopy scan. Default is 1000."""

    frequency_span_in_mhz: float = 10.0
    """Span of frequencies to sweep in MHz. Default is 10 MHz."""

    frequency_step_in_mhz: float = 0.05
    """Step size for frequency sweep in MHz. Default is 0.05 MHz."""

    probe_amplitude_scale: float = 0.1
    """Probe power, as a scale factor on each qubit's operating readout amplitude. Kept well below 1 so
    that the resonator is probed in the low-power (unsaturated) regime. Must be within [0, 2). Default is 0.1."""

    probe_skip_ns: int | None = None
    """Start of the probe pulse that is not integrated, while the resonator is still filling up. The circle
    fit assumes the steady-state response; integrating the fill-up broadens the resonance and biases kappa
    high (about +90% for a 0.8 us readout-length probe at kappa/2pi = 0.7 MHz). None derives it per qubit as
    8 / kappa (fill-up transient below ~2%) from the stored linewidth, or 2000 ns when none is stored.
    Multiple of 4."""

    probe_window_ns: int | None = None
    """Integration window after the skip. Longer averages more signal, but the qubit prepared in |1> decays
    during it, which mixes the two resonances and inflates the |1> linewidth. None takes min(2000 ns, T1 / 10)
    per qubit (2000 ns when T1 is unknown). Multiple of 4."""

    measure_excited_state: bool = True
    """Also sweep the resonance with the qubit prepared in |1>, which is what yields chi. The two
    states are measured back to back at each frequency, so slow drift subtracts out of the splitting.
    It doubles the acquisition time. Set to False to get the linewidth alone. Default is True."""

    fit_cable_delay: bool = True
    """Fit the cable delay from the measured phase slope. When False, the stored `time_of_flight` is used
    as-is, which makes Q_c unreliable; leave True unless debugging. Default is True."""

    min_r_squared: float = 0.8
    """Minimum R² of the fitted |S21| magnitude against the data for the fit to count as successful."""

    max_chi_over_kappa: float = 0.3
    """Warn when |chi| / kappa exceeds this. Above it the weak-dispersive formulas that node 25 uses
    to turn a Stark shift into a photon number no longer hold, and the general steady-state
    expressions are needed instead. Default is 0.3."""

    chi_mismatch_warning_fraction: float = 0.2
    """Warn when the fitted chi disagrees with a chi already stored on the qubit by more than this
    fraction. Everything node 25 reports scales with chi. Default is 0.2, i.e. 20%."""

    kappa_mismatch_warning_fraction: float = 0.2
    """Warn when the fitted kappa_tot disagrees with a kappa already stored on the resonator by more
    than this fraction. Default is 0.2, i.e. 20%."""

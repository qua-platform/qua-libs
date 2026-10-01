from typing import List, Optional

from qualibrate import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameter class for the AC Stark photon calibration node."""

    num_shots: int = 1000
    """Number of averages to perform. Default is 1000."""

    free_evolution_times_in_ns: List[int] = [20, 40, 60, 90, 120, 160, 240, 400]
    """Free evolution times of the Ramsey sequence, in ns, swept as an outer axis.

    The Stark shift is the slope of the accumulated fringe phase against this axis, so a phase that
    does not grow in proportion to the time the tone was on is visible rather than silent. The times
    must be short enough that the fringe still lives at the largest tone amplitude: the contrast
    falls as exp(-Gamma_d * tau), and on a resonator with |chi| comparable to kappa the fringe has
    halved by the time the Stark phase reaches about 0.7 rad whatever amplitude is used, so the top
    of the amplitude sweep only ever contributes the shortest times.

    The range has to reach past the resonator photon lifetime, not just span a factor of a few. Over
    times all shorter than 1/kappa a resonator transient grows almost linearly too, so a phase that
    is really a transient still fits a straight line and the linearity check has nothing to catch it
    on. The long times survive at the bottom of the amplitude sweep, where the dephasing is weak, and
    that is where the check gets its leverage. Default is six times from 20 to 1000 ns, against a
    photon lifetime of about 190 ns on a 850 kHz resonator.

    Times past the photon-number-splitting limit, where 2 chi tau approaches a radian, are dropped
    per qubit, so the list can carry longer times for the sake of a chip with a smaller chi without
    corrupting one with a large chi. On a chip where that limit falls below the photon lifetime the
    two requirements cannot both be met, and the node says so."""

    steady_state_pad_in_lifetimes: float = 8.0
    """Length of the tone held before the first pi/2 pulse, in resonator photon lifetimes 1/kappa.

    The resonator fills with a field time constant of 2/kappa, so without a pad the photon number
    rises throughout the free evolution and the measured shift is not a property of the drive at
    at all. Eight lifetimes reach 96% of the steady-state photon number; two reach only
    40%. Set to zero to reproduce the unpadded ring-up deliberately. Default is 8."""

    depletion_in_lifetimes: float = 8.0
    """Time between the second pi/2 pulse and the readout, in resonator photon lifetimes.

    The tone stops with the second pi/2, and the resonator has to empty before the readout starts or
    the discrimination is biased by photons left over from the tone. The qubit state is already in
    populations by then, so this wait costs fidelity nothing. The larger of this and the stored
    `resonator.depletion_time` is used. Default is 8."""

    num_amplitude_points: int = 11
    """Number of Stark tone amplitude points, including the zero-amplitude reference. Default is 11."""

    max_amplitude_scale: float = 0.3
    """Highest tone amplitude, as a scale on the qubit's operating readout amplitude. Scale 1.0 is the
    operating amplitude. Must stay below 2, which is the hardware limit on a real-time amplitude
    scale. The useful ceiling is set by the fringe rather than by the hardware: measurement-induced
    dephasing kills the fringe well before the Stark shift becomes comparable to the pi/2 Rabi rate,
    so the photon number at the operating amplitude is reached by extrapolating the fitted line.
    Default is 0.3."""

    num_phase_points: int = 21
    """Number of phase points of the second pi/2 pulse. The phases span exactly one full turn.
    Default is 21."""

    tone_frequency_in_ghz: Optional[float] = None
    """Absolute frequency of the Stark tone in GHz. When left as None the tone sits at the qubit's
    current readout frequency, so the calibration applies to the conditions the qubit is actually
    read out in. Default is None."""

    separate_resonator_core: bool = True
    """Give the resonator its own pulse-processor core for the duration of the run.

    Elements that share a core cannot play overlapping pulses, so a resonator sharing a core with
    its qubit's drive cannot hold a tone across the Ramsey sequence: the tone is played first and
    the Ramsey follows in the ring-down, which produces a fringe phase that does not grow with the
    free evolution time. The change is made through `tracked_updates` and reverted when the node
    finishes. It costs one extra core per targeted qubit. Default is True."""

    min_abs_chi_in_hz: float = 1e4
    """A qubit whose stored |chi| is below this is skipped with a logged reason rather than producing
    a photon number divided by an implausible dispersive shift. Default is 10 kHz."""

    max_abs_chi_in_hz: float = 5e7
    """A qubit whose stored |chi| is above this is skipped with a logged reason. Default is 50 MHz."""

    min_contrast_fraction: float = 0.15
    """Drop, at each amplitude, the free evolution times whose fringe contrast has fallen below this
    fraction of the zero-amplitude contrast at the same time. A collapsed fringe carries no phase,
    and fitting one returns a number drawn from the noise. Default is 0.15."""

    max_number_splitting_phase_rad: float = 1.0
    """Drop free evolution times at which 2 chi tau exceeds this, in radians.

    The Stark shift is 2 chi n_bar only while the qubit cannot resolve one photon from the next. Once
    2 chi tau approaches a radian the qubit starts to resolve photon number and the apparent fringe
    phase follows n_bar sin(2 chi tau), which stops growing and then runs backwards past 2 chi tau of
    pi/2. A straight line fitted through those times returns a photon number that is simply too
    small. At chi = 460 kHz the limit of 1 rad falls at 173 ns. Default is 1.0."""

    min_tau_over_photon_lifetime: float = 2.0
    """Warn when no amplitude keeps a usable fringe past this many resonator photon lifetimes.

    Below about one lifetime a resonator transient and a true Stark phase both grow almost linearly
    with the free evolution time, so the linearity check cannot separate them and a straight line
    proves nothing. Default is 2."""

    min_phase_linearity_r_squared: float = 0.9
    """Minimum R² of the accumulated fringe phase against the free evolution time for the qubit to
    count as successful. This is the check that a single free evolution time could not make: a phase
    that is really Delta_omega times tau grows in proportion to tau, and one that does not is not a
    Stark phase whatever else fits. Default is 0.9."""

    min_linear_fit_r_squared: float = 0.9
    """Minimum R² of the photon number against tone power for the calibration to count as
    successful. Default is 0.9."""

    phase_step_warning_fraction_of_pi: float = 0.8
    """Warn when the fringe phase grows by more than this fraction of pi between adjacent free
    evolution times. The phase is unwrapped along the time axis, which holds only while the step
    stays below pi; times spread too far apart alias. Default is 0.8."""

    gamma_ratio_warning_fraction: float = 0.2
    """Warn when the measured Gamma_d / Delta_omega disagrees with the value predicted from chi,
    kappa and the tone detuning by more than this fraction. The prediction contains neither the
    photon number nor the drive amplitude, so a disagreement means the tone is not where it is
    assumed to be, or something other than measurement backaction is decohering the qubit.
    Default is 0.2, i.e. 20%."""

    n_bar_gamma_warning_fraction: float = 0.3
    """Warn when the photon number derived from the induced dephasing disagrees with the one derived
    from the Stark phase by more than this fraction. The two come from the same fringes by different
    routes, so this is the same check as the ratio above expressed in the unit the node exists to
    report. Default is 0.3."""

    weak_dispersive_warning_ratio: float = 0.3
    """Warn when |chi| / kappa exceeds this. The formulas turning a Stark shift into a photon number
    assume the weak-dispersive limit, and above about 0.3 the two dressed resonances are separated by
    less than a linewidth. Default is 0.3."""

    min_plausible_coupling_g_in_hz: float = 1e7
    """Warn when the coupling g implied by the stored chi, detuning and anharmonicity falls below
    this. g is not measured here, so an implausible value means one of those three is wrong, and the
    critical photon number derived from them with it. Default is 10 MHz."""

    max_plausible_coupling_g_in_hz: float = 5e8
    """Warn when the implied coupling g rises above this. Default is 500 MHz."""

    n_bar_over_critical_warning_fraction: float = 0.5
    """Warn when the measured photon number at the operating readout amplitude exceeds this fraction
    of the critical photon number, i.e. when the readout is running close to the power at which the
    dispersive approximation breaks down. Default is 0.5."""

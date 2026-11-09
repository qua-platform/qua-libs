from typing import List, Literal, Optional
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 20
    """Number of averages to perform. Default is 100."""
    qubit_initial_states: List[Literal["g", "e"]] = ["g", "e"]
    """The states the qubit is prepared in before the resonator is driven. The ac Stark shift Lorentzian is centred
    on f_r - chi for "g" and f_r + chi for "e", so both are needed to get chi, kappa and the drive power out of
    Eq. (7) of PRApplied 23, 024055. The two states are interleaved within each shot, so slow drifts shift both
    curves alike and largely cancel in their separation. Pass a single state to acquire only that panel, in which
    case the dressed resonator frequency and kappa are still reported but chi is not. Default is ["g", "e"]."""

    # --- ac Stark drive applied to the readout resonator (swept axis #1) ---
    resonator_frequency_span_in_mhz: float = 6.0
    """Span of the resonator drive frequency sweep, centred on the readout frequency, in MHz. Default is 6 MHz,
    which covers several linewidths for a resonator of a fraction of a MHz. Fig. 2 of PRApplied 23, 024055 uses
    +/- 10 MHz, i.e. a span of 20 MHz; widen it if the readout frequency is only crudely known, bearing in mind
    that every extra point costs a full qubit reset."""
    resonator_frequency_step_in_mhz: float = 0.06
    """Step of the resonator drive frequency sweep in MHz. Default is 0.06 MHz."""
    resonator_drive_operation: str = "const"
    """The resonator operation used to populate the resonator. Must be a square pulse, since its duration is
    overridden at runtime. Default is "const"."""
    resonator_drive_amplitude_factor: float = 1.0
    """Amplitude pre-factor for the resonator drive, restricted to [-2; 2). Raise it until the ac Stark shift on
    resonance is a few times the qubit probe linewidth (-20 MHz is used in the paper). Default is 1.0."""
    resonator_drive_amplitude_in_v: Optional[float] = None
    """Absolute amplitude of the resonator drive operation in V, applied as a temporary override of the operation
    defined in the state and reverted at the end of the node. Use it when the [-2; 2) range of
    `resonator_drive_amplitude_factor` does not reach a large enough ac Stark shift. Default is None, which leaves
    the operation untouched."""
    separate_drive_and_probe_cores: bool = True
    """Whether to temporarily give the xy drive a core of its own. The qubit probe has to be played while the
    resonator is still being driven, but elements sharing a core are serialised by the pulse processor, so with a
    shared core the probe is pushed to after the drive and no ac Stark shift is measured. Only set this to False
    if the xy drive and the resonator already sit on different cores, or if the core budget of the controller is
    exhausted. Default is True."""
    resonator_drive_settle_time_in_ns: int = 1000
    """Time the resonator is driven before the qubit is probed, in ns. Must be much longer than 1/kappa so that the
    resonator has reached its steady-state photon occupation. Default is 1000 ns."""

    # --- Qubit probe tone (swept axis #2) ---
    qubit_detuning_min_in_mhz: float = -35.0
    """Lower end of the qubit probe detuning sweep in MHz. The sweep is deliberately asymmetric because the ac Stark
    shift 2 * chi * n_bar pushes the qubit towards negative detunings when chi < 0. Default is -40 MHz."""
    qubit_detuning_max_in_mhz: float = 5.0
    """Upper end of the qubit probe detuning sweep in MHz. Default is 5 MHz."""
    qubit_detuning_step_in_mhz: float = 0.1
    """Step of the qubit probe detuning sweep in MHz. Default is 0.5 MHz, since a 100 ns probe is power broadened to
    several MHz and a finer step only lengthens the node."""
    qubit_probe_operation: str = "saturation"
    """The qubit operation used to probe the ac Stark shifted qubit frequency. Must be a square pulse, since its
    duration is overridden at runtime. Default is "saturation"."""
    qubit_probe_len_in_ns: int = 100
    """Duration of the qubit probe pulse in ns. Default is 100 ns, as used in the paper."""
    qubit_probe_amplitude_factor: Optional[float] = None
    """Amplitude pre-factor for the qubit probe pulse, restricted to [-2; 2). If None, it is derived from the
    calibrated x180 pulse so that the probe has a pulse area of `qubit_probe_pi_fraction` times a pi pulse.
    Default is None."""
    qubit_probe_pi_fraction: float = 0.8
    """Pulse area of the qubit probe, in units of the area of a pi pulse, used when `qubit_probe_amplitude_factor`
    is None. Default is 0.8, as used in the paper."""

    # --- Analysis ---
    max_clipped_fraction: float = 0.05
    """Largest fraction of resonator drive frequencies whose ac Stark shift may sit at the edge of the qubit probe
    window before the fit is declared failed. Beyond this the Lorentzian is clipped, because the drive is strong
    enough to push the qubit past the swept window, and its depth and width are meaningless. Default is 0.05."""
    max_relative_residual: float = 0.25
    """Largest RMS residual of the ac Stark shift Lorentzian fit, in units of the fitted depth, before the fit is
    declared failed. A large residual means the Lorentzian does not describe the data, which typically happens
    when the drive is strong enough to push the transmon out of the computational subspace near resonance and
    leave a saturated band of noise. Default is 0.25."""
    min_points_per_linewidth: float = 5.0
    """Smallest number of resonator drive frequencies that must fall across the fitted linewidth before the fit is
    declared failed. kappa is read off the width of the Lorentzian, so a drive step comparable with kappa leaves
    only a couple of points on the feature and the fitted width is set by the sampling rather than by the
    resonator. Narrow 'resonator_frequency_span_in_mhz' and shrink 'resonator_frequency_step_in_mhz' instead of
    lowering this. Default is 5.0."""
    peak_prominence_factor: float = 3.0
    """How prominent the qubit peak must be, in units of the noise standard deviation, to be picked up at a given
    resonator drive frequency. Default is 3.0."""
    update_state: bool = False
    """Whether to store the fitted linewidth and dressed resonator frequency in qubit.resonator.extras, under the
    keys already used by the state ("kappa_hz"/"f_01_ground" for "g" and "kappa_e_hz"/"f_01_excited" for "e"), and
    to update qubit.chi from the separation of the two dressed resonator frequencies. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Resonator spectroscopy specific parameters."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""

    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""

    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""

    # --- v2 fit gates ---
    min_dip_snr: float = 6.0
    """FREQUENCY gate: minimum dip significance (baseline-subtracted prominence / per-point noise sigma) to count the resonator as found; only R²/FWHM/contrast (not this) gate `success_shape`."""

    dip_dominance: float = 2.0
    """Flags a window `ambiguous` when the second-most-prominent dip is within this factor of the top one, e.g. wide bring-up scans catching feedline neighbours; verify against expected frequency or vs-power punch-out."""

    # --- Bring-up span escalation (no-dip retry) ---
    escalate_on_no_dip: bool = False
    """When True, re-measures qubits with no significant dip using a doubled span (up to `max_escalation_span_in_mhz`), re-centering the readout LO as needed; fresh-from-fab bring-up only, leave False for routine tracking."""

    max_escalation_span_in_mhz: float = 800.0
    """Span ceiling (MHz) for the no-dip escalation ladder. Default ±400 MHz."""

    # --- Optional diagnostic plots (off by default; amplitude+fit and detrended
    # phase are shown unconditionally) ---
    show_raw_phase_plot: bool = False
    """Show the raw phase + group delay figure; useful when the amplitude dip is weak/ambiguous and phase gives a sharper resonance feature."""

    show_iq_circle_plot: bool = False
    """Show the I/Q parametric trace figure; a readout-troubleshooting tool (impedance mismatch, weak coupling, mixer issues), not needed for routine fits."""

    # --- Re-fit overrides (used together with load_data_id) ---
    re_fit_resonators: list[str] | None = None
    """Qubit names to re-fit with a manually specified window, e.g. ["qA1", "qD3"]; must be the same length as re_fit_centers_ghz and re_fit_span_mhz."""

    re_fit_centers_ghz: list[float] | None = None
    """Absolute RF center frequency (GHz) for each qubit in re_fit_resonators."""

    re_fit_span_mhz: list[float] | None = None
    """Fit span (MHz) around the center for each qubit in re_fit_resonators."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

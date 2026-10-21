"""Parameters for three-tone coupler spectroscopy vs coupler flux (22b)."""

from typing import ClassVar, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Three-tone 2D spectroscopy sweeping coupler flux-pulse amplitude and drive frequency."""

    num_shots: int = 100
    """Number of averages per (flux, frequency) point."""
    control_drive_operation: str = "x180"
    """Control-qubit operation used to drive the coupler."""
    control_pulse_duration_in_ns: int = 500
    """Duration of the control drive pulse in ns."""
    control_pulse_amplitude: float = 0.1
    """Amplitude scale for the control drive pulse."""
    target_drive_operation: str = "saturation"
    """Weak probe operation on the target qubit."""
    target_pulse_amplitude: float = 0.02
    """Amplitude scale for the target probe pulse."""
    target_pulse_duration_in_ns: Optional[int] = 400
    """Target probe duration in ns; default uses the operation length from state."""
    frequency_span_in_mhz: float = 800.0
    """Total frequency span of the coupler drive sweep in MHz."""
    frequency_step_in_mhz: float = 1.0
    """Frequency step in MHz."""
    coupler_flux_min_in_v: float = -0.025
    """Minimum coupler flux-pulse amplitude in V (played relative to decouple_offset)."""
    coupler_flux_max_in_v: float = 0.025
    """Maximum coupler flux-pulse amplitude in V (played relative to decouple_offset)."""
    num_coupler_flux_points: int = 11
    """Number of coupler flux-pulse amplitudes."""
    coupler_flux_settle_in_ns: int = 25
    """Wait after the coupler flux pulse starts, before the control drive, in ns."""
    coupler_band: Literal["above", "below"] = "above"
    """Where the coupler sits relative to the qubit pair (used only for the RF guess).

    ``above``: idle above the higher qubit (BAQ). ``below``: idle below the lower qubit (BBQ).
    Ignored when ``rf_frequency_startpoint_in_hz`` or ``coupler.RF_frequency`` is set.
    """
    coupler_idle_detuning_in_ghz: float = 1.2
    """Idle detuning from the nearest qubit used for the RF guess, in GHz.

    Typical literature parks are ~1–1.5 GHz away. ``above`` uses ``max(f) + detuning``;
    ``below`` uses ``min(f) - detuning``. Ignored when
    ``rf_frequency_startpoint_in_hz`` or ``coupler.RF_frequency`` is set.
    """
    rf_frequency_startpoint_in_hz: Optional[float] = None
    """Optional coupler RF sweep centre in Hz for all pairs.

    When ``None`` (default), each pair uses ``coupler.RF_frequency`` from state if set,
    otherwise an estimate from the qubit XY frequencies and ``coupler_band``.
    """


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for 22b three-tone coupler spectroscopy vs flux."""

    targets_name: ClassVar[str] = "qubit_pairs"

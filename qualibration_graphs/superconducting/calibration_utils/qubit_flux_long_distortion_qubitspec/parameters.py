"""Parameter definitions for pi vs flux (long time distortion) calibration experiment."""

from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for long flux distortion characterization."""

    num_shots: int = 30
    """Number of shots to acquire."""
    operation: str = "x180"
    """Operation to excite the qubit"""
    operation_amplitude_factor: float = 1.0
    """Amplitude factor for the operation."""
    duration_in_ns: int = 8000
    """Maximum duration of the sequence."""
    time_axis: Literal["linear", "log"] = "log"
    """Time axis for the operation."""
    time_step_in_ns: int = 48
    """Time step in nanoseconds. For linear time axis."""
    time_step_num: int = 100
    """Number of time steps. Used for log time axis."""
    min_wait_time_in_ns: int = 32
    """Minimum wait time in nanoseconds."""
    detuning_in_mhz: float = 200.0
    """Target detuning below idle frequency in MHz (positive = below idle); spectroscopy sweep is centered at (idle - detuning). Default is 200.0."""
    frequency_span_in_mhz: float = 200.0
    """Total frequency sweep width in MHz centered on the detuning point; covers [detuning - span/2, detuning + span/2]. Default is 200.0."""
    frequency_step_in_mhz: float = 1.0
    """Frequency step in MHz for the uniform spectroscopy sweep passed to QUA from_array. Default is 1.0."""
    flux_branch: Literal["left", "right"] = "right"
    """Which branch of the freq-vs-flux curve to use (as seen on the spectroscopy plot). e.g. 'right' = flux >= idle flux (positive flux direction from sweetspot)."""
    n_exponentials: int = 2
    """Number of exponential components to fit in the flux step response model."""
    update_state: bool = False
    """Whether to update the state. CAUTION: assumes fitting will be acceptable"""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI, select when fitting is successful"""
    buffer_during_operation_in_ns: int = 800
    """Buffer time in ns during the operation to avoid turn-off transient overlapping with the XY pulse."""
    buffer_after_operation_in_ns: int = 800
    """Buffer time in ns after the operation to keep readout clean from the flux turn-off artifact."""
    use_spectroscopy_data: bool = False
    """When True, load raw qubit spectroscopy vs flux from spectroscopy_run_id and extract the freq-vs-flux curve, instead of using freq_vs_flux_01_quad_term."""
    spectroscopy_run_id: Optional[int] = None
    """Run ID of a previous ``03b_qubit_spectroscopy_vs_flux`` experiment."""
    use_ramsey_data: bool = False
    """When True, load Ramsey vs Z-flux data for freq->flux conversion (priority-2 path after spectroscopy, before quad_term fallback)."""
    ramsey_run_id: Optional[int] = None
    """Run ID of a previous ``09a_ramsey_vs_flux_calibration`` experiment. If None (default), the ID is read automatically from each qubit's extras['ramsey_vs_flux_calibration_load_id']."""
    debug_plots: bool = False
    """If True, also generate diagnostic figures (IQ/phase heatmaps, spectroscopy/Ramsey reference curves)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for pi vs flux calibration node."""

    pass

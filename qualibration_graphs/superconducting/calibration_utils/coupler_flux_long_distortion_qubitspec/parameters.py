"""Parameter definitions for coupler flux long distortion calibration (qubitspec variant)."""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Specific parameters for coupler flux long distortion characterization."""

    num_shots: int = 30
    """Number of shots to acquire."""
    operation: str = "x180"
    """Operation to excite the qubit."""
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
    detuning_in_mhz: float = -5.0
    """Signed detuning from the qubit frequency at the coupler's decouple_offset in MHz (positive = above, negative = below); reference frequency is read from the loaded dispersion curve at decouple_offset (state.json). Default is -5.0."""
    frequency_span_in_mhz: float = 10.0
    """Total frequency sweep width in MHz centered on the detuning point; covers [detuning - span/2, detuning + span/2]. Default is 10.0."""
    frequency_step_in_mhz: float = 0.5
    """Frequency step in MHz for the uniform spectroscopy sweep passed to QUA from_array. Default is 0.5."""
    n_exponentials: int = 3
    """Number of exponential components to fit in the flux step response model."""
    update_state: bool = False
    """Master gate for writing fitted filters into QUAM state."""
    update_state_from_GUI: bool = False
    """When re-analysing via ``load_data_id``, enable ``update_state`` from the GUI."""
    debug_plots: bool = False
    """If True, also show diagnostic probe figures (spectroscopy heatmap or center
    frequency trace, and the coupler dispersion curve used for freq→flux inversion).

    Default figure (always): flux response vs time with the IIR fit overlay.
    """
    coupler_flux_amplitude_in_v: float = 0.1
    """Fallback coupler flux amplitude in V when no dispersion curve is available in state."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit to measure: 'control' or 'target'. Default is 'target'."""
    buffer_during_operation_in_ns: int = 600
    """Buffer time in ns during the operation to avoid turn-off transient overlapping with the XY pulse."""
    buffer_after_operation_in_ns: int = 600
    """Buffer time in ns after the operation to keep readout clean from the flux turn-off artifact."""
    freq_to_flux_source: Literal["auto", "spectroscopy", "ramsey"] = "auto"
    """Which frequency→coupler-flux relation to use for amplitude derivation and analysis.
    ``auto`` (default) tries 03c spectroscopy, then 09b Ramsey.
    Run IDs are read from qubit extras — run 03c / 09b with ``save_load_id=True`` first."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for coupler flux long distortion calibration (dataload variant)."""

    targets_name: ClassVar[str] = "qubit_pairs"

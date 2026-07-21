from typing import Literal, Optional
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots to acquire per detuning point. Default is 100."""
    qubit_pair_to_initialize: Optional[str] = None
    """Initialize the qubit pair. If None, it will default to the same pair as the qubit pair for measurement."""
    qubit_to_pulse: Optional[str] = None
    """Optionally apply a pi pulse to the qubit."""
    barrier_gate_voltage: float = 0.0
    """Barrier Gate Voltage to pulse to with the detuning. Default zero."""
    detuning_min: float = -0.1
    """Minimum detuning value for the sweep in volts. Default is -0.1 V."""
    detuning_max: float = 0.1
    """Maximum detuning value for the sweep in volts. Default is 0.1 V."""
    detuning_points: int = 21
    """Number of detuning points to sweep. Default is 21."""
    ramp_duration: int = 40
    """Ramp duration to ramp to the measurement point."""
    buffer_duration: int = 16
    """Buffer duration at the measurement point before readout pulse."""
    initialization_macro: Literal["empty", "initialize"] = "empty"
    """Which dot-pair macro runs for the preparation step (formerly ``dot_pair.initialize()``).
    Both ``empty`` and ``initialize`` must exist on ``dot_pair.macros``."""

    # ----- iq_sweep analysis fields -----
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of resonator operation whose readout parameters are optimised. Default "readout"."""
    sweep_name: str = "detuning"
    """Name of the swept coordinate in ds_raw (fixed to "detuning" here but kept
    explicit so iq_sweep analysis remains generic)."""
    optimization_metric: Literal["fidelity", "visibility"] = "fidelity"
    """Metric used to pick the optimal detuning for state updates.
    Both fidelity and visibility optima are recorded regardless of this choice."""
    labeled_states: bool = False
    """Whether ds_raw contains labelled S/T preparations (Ig,Qg,Ie,Qe) or a
    single mixed-state acquisition (I,Q). PSB search uses random loading, so
    defaults to False. Set True only if you explicitly prepare S and T."""
    use_simulated_data: bool = False
    """If True, skip QUA compile/execute and build synthetic shot-by-shot I/Q
    (Barthel-style forward model) for offline analysis. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    pass

from typing import Literal, Optional
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters
from calibration_utils.iq_utils import IQSweepParameters


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
    use_simulated_data: bool = False
    """If True, skip QUA compile/execute and build synthetic shot-by-shot I/Q
    (Barthel-style forward model) for offline analysis. Default False."""


class Parameters(
    NodeParameters, CommonNodeParameters, NodeSpecificParameters, QubitPairExperimentNodeParameters, IQSweepParameters
):
    pass

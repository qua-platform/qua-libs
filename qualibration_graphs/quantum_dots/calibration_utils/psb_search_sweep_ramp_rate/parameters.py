from typing import Literal, Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters
from calibration_utils.iq_utils import IQSweepParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots to acquire per ramp-duration point. Default is 100."""
    ramp_duration_min: int = 16
    """Minimum ramp duration to the measure point (ns). Must be a multiple of 4."""
    ramp_duration_max: int = 400
    """Maximum ramp duration (ns); sweep uses ``np.arange(min, max, step)``. Must be a multiple of 4."""
    ramp_duration_step: int = 16
    """Step between ramp durations (ns). Must be a multiple of 4."""
    buffer_duration: int = 16
    """Hold duration at the measurement point before readout (ns)."""
    reset_wait_time: int = 16
    """Settling time at zero volts between consecutive sweep points (ns)."""
    detuning: Optional[float] = None
    """If set, temporarily overrides the measure macro detuning voltage (V) for this node only."""
    initialization_macro: Literal["empty", "initialize"] = "empty"
    """Which dot-pair macro runs for the preparation step (formerly ``dot_pair.initialize()``).
    Both ``empty`` and ``initialize`` must exist on ``dot_pair.macros``."""
    use_simulated_data: bool = False
    """If True, skip QUA compile/execute and build synthetic shot-by-shot I/Q. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitPairExperimentNodeParameters,
    IQSweepParameters,
    NodeSpecificParameters,
):
    sweep_name: str = "ramp_duration"
    """Name of the swept coordinate in ``ds_raw`` (ramp duration in ns)."""

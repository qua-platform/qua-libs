from typing import Literal, Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters
from calibration_utils.iq_utils import IQSweepParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots to acquire per readout-length point. Default is 100."""
    readout_length_min: int = 100
    """Minimum readout pulse length (nanoseconds). Default is 100."""
    readout_length_max: Optional[int] = None
    """If set, upper bound (ns) for the sweep; the compiled pulse is shortened to the largest
    value ``N * 4 * segment_length`` not exceeding this (QM integration-weight constraint).
    If None, the first pair's current ``readout`` operation length from QUAM is used."""
    readout_length_points: int = 100
    """Target number of sweep steps; actual count follows the same arange rule as charge-state readout time (step in ns, multiples of 4). Default is 21."""
    ramp_duration: int = 40
    """Ramp duration to ramp to the measurement point."""
    buffer_duration: int = 16
    """Buffer duration at the measurement point before readout pulse."""
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
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
    IQSweepParameters,
):
    pass

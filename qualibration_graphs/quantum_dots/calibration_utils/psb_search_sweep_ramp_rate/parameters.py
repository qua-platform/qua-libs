from typing import Literal, Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from calibration_utils.common_utils.experiment import QubitPairExperimentNodeParameters
from qualibration_libs.parameters import CommonNodeParameters


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
    detuning: Optional[float] = None
    """If set, temporarily overrides the measure macro detuning voltage (V) for this node only."""
    initialization_macro: Literal["empty", "initialize"] = "empty"
    """Which dot-pair macro runs for the preparation step (formerly ``dot_pair.initialize()``).
    Both ``empty`` and ``initialize`` must exist on ``dot_pair.macros``."""

    # ----- iq_sweep analysis fields -----
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of resonator operation whose readout parameters are optimised. Default \"readout\"."""
    sweep_name: str = "ramp_duration"
    """Name of the swept coordinate in ds_raw (ramp duration in ns). Shorter duration implies a
    higher effective ramp rate for the same voltage trajectory."""
    optimization_metric: Literal["fidelity", "visibility"] = "fidelity"
    """Metric used to pick the optimal ramp duration for state updates."""
    labeled_states: bool = False
    """PSB search uses random loading; defaults to False."""
    use_simulated_data: bool = False
    """If True, skip QUA compile/execute and build synthetic shot-by-shot I/Q. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    pass

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QuantumDotExperimentNodeParameters

from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: Optional[List[str]] = None
    """List of sensor names to include in the measurement. If None, all sensors are used."""
    v_center: float = 0.0
    """Center of the voltage sweep in volts. Default 0.0V."""
    v_span: float = 0.01
    """Span of the voltage sweep in volts. Default 10mV."""
    num_points: int = 101
    """Number of points in the voltage sweep. Default is 101."""
    dc_control: bool = False
    """If checked, then v_center will be applied by your external source via the VirtualDCSet. If not, then v_center will be applied by the OPX."""
    dwell_time: int = 500
    """Dwell time at each voltage in nanoseconds. Default is 500ns."""


class Parameters(
    NodeParameters,
    QuantumDotExperimentNodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

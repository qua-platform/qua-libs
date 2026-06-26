from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import BaseExperimentNodeParameters
from typing import Optional, List


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    qubit_pair: List[str] = ["q1_q2"]
    """Qubit pair to measure."""
    offset_span: float = 0.1
    """Minimum voltage offset for the sensor gate sweep in volts. Default is -0.2 V."""
    offset_step: float = 0.01
    """Step size for the voltage offset sweep in volts. Default is 0.005 V."""
    duration_after_step: int = 1000
    """Wait duration after each voltage step in nanoseconds. Default is 1000 ns (1 µs)."""
    dac_settling_time_s: float = 1
    """Wait duration after setting the DAC voltage. Done in Python, not QUA."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be used to measure."""
    dac_sensor_name: str = None
    """The name of the DAC sensor to sweep."""
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
):
    pass

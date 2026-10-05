from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters
from typing import Optional, List, Literal


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    offset_min: float = -0.2
    """Minimum voltage offset for the sensor gate sweep in volts. Default is -0.2 V."""
    offset_max: float = 0.2
    """Maximum voltage offset for the sensor gate sweep in volts. Default is 0.2 V."""
    offset_step: float = 0.01
    """Step size for the voltage offset sweep in volts. Default is 0.01 V."""
    duration_after_step: int = 1000
    """Wait duration after each voltage step in nanoseconds. Default is 1000 ns (1 µs)."""
    sensor_names: Optional[List[str]] = None
    """The list of sensor dot names to be included in the measurement. """
    use_simulated_data: bool = False
    """Whether to generate simulated data instead of measuring via the OPX. Default False."""
    peak_fit_side: Literal["left", "right"] = "left"
    """Which side to fit the max gradient on."""
    max_compensation_voltage: float = 0.01
    """The maximum compensation pulse voltage."""
    ramp_duration: int = 16
    """Ramp duration of each voltage point."""
    qubit_pair_to_step: List[str] | None = None
    """Qubit pair to step to the measure point for OPX, during readout. Default to None, which will not step anything."""
    dac_settling_time_s: float = 0.5
    """Wait duration after setting the DAC voltage. Done in Python, not QUA."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    BaseExperimentNodeParameters,
):
    pass

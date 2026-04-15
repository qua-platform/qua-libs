from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QuantumDotExperimentNodeParameters

from typing import List, Optional


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    sensor_names: List[str]
    """List of sensor dot names used for measurement."""
    components: Optional[List[str]]
    """Components (quantum dots, barriers, sensors, etc) to calibrate."""
    square_wave_frequency: int = 2e6
    """Frequency of the square wave sent through the fast line. Ensure that this is sufficiently high relative to the bias tee cut-off."""
    square_wave_amplitude: float = 0.001
    """Amplitude of the square wave sent through the fast line."""
    dc_sweep_span: float = 0.01
    """The amplitude of the DC sweep around the current point"""
    dc_sweep_step: float = 0.0005
    """The DC sweep step size"""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QuantumDotExperimentNodeParameters,
    NodeSpecificParameters,
):
    pass

from typing import Optional
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a qubit spectroscopy experiment.

    Attributes:
        num_shots (int): Number of averages to perform. Default is 100.
        operation (str): Type of operation to perform. Default is "saturation".
        min_amp_factor (float): Minimum amplitude pre-factor for sweeping the DRAG coefficient. Default is -1.0.
        max_amp_factor (float): Maximum amplitude pre-factor for sweeping the DRAG coefficient. Default is 2.0.
        amp_factor_step (float): Step of amplitude pre-factor for sweeping the DRAG coefficient. Default is 0.02.
        max_number_pulses_per_sweep (int): Maximum number of drive pulses. Default is 40.
        alpha_setpoint (Optional[float]): Optional setpoint for the alpha coefficient. Default is None.
    """

    num_shots: int = 10
    operation: str = "x180"
    min_amp_factor: float = -1
    max_amp_factor: float = 2.0
    amp_factor_step: float = 0.02
    max_number_pulses_per_sweep: int = 40
    alpha_setpoint: Optional[float] = None


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters

from typing import List

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""

    virtual_gates_id: str = None
    x_axis_name: str = None
    y_axis_name: str = None
    sensor_names: List[str] = []
    dc_control: bool = False
    result_type: str = "I"

    



class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

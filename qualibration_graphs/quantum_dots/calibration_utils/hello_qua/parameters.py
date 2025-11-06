from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from typing import List

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    v_span: float = 0.01
    """Span of 1d sweep"""
    num_points: int = 51
    """Resolution of 1d sweep"""

    quantum_dots: List[str] = None
    sensor_names: List[str] = None

    virtual_gate_set_id: str = None

    Video_Mode: bool = True
    dc_control: bool = False




class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

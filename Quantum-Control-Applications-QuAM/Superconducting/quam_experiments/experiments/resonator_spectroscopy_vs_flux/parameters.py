from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a # todo ... experiment.

    Attributes:
        # todo: param (type): description. Default
        num_averages (int): Number of averages to perform. Default is 100.
    """
    num_averages: int = 100
    min_flux_offset_in_v: float = -0.5
    max_flux_offset_in_v: float = 0.5
    num_flux_points: int = 101
    frequency_span_in_mhz: float = 15
    frequency_step_in_mhz: float = 0.1
    input_line_impedance_in_ohm: float = 50
    line_attenuation_in_db: float = 0
    update_flux_min: bool = False


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass

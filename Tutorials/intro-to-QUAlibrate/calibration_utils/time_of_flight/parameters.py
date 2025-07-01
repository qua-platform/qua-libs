"""
Node parameter definitions for the time-of-flight experiment.

This module defines the configurable parameters used in the node.
"""

from typing import List
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters specific to this node's time-of-flight acquisition logic.

    Attributes:
        num_shots (int): Number of averages to perform. Default is 100.
        depletion_time (int): Time in ns to wait for resonator depletion. Default is 10000.
        simulate (bool): If True, simulate the experiment; if False, execute on hardware. Default is False.
        resonators (List[str]): List of resonator identifiers to measure. Default is ["q1_resonator"].
        multiplexed (bool): If True, perform multiplexed readout. If False, perform parallel readout. Default is True.
    """

    num_shots: int = 100
    depletion_time: int = 10000
    simulate: bool = True
    resonators: List[str] = ["q1_resonator"]
    multiplexed: bool = True


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass

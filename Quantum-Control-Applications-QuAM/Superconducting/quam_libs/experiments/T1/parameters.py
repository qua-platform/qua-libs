from typing import List, Literal, Optional

import numpy as np
from qualibrate import NodeParameters

from quam_libs.experiments.node_parameters import (
    QubitsExperimentNodeParameters,
    SimulatableNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters
)


class Parameters(
    NodeParameters,
    QubitsExperimentNodeParameters,
    SimulatableNodeParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters
):
    
    """
    Parameters for the T1 experiment.
    Attributes:
        num_averages (int): Number of averages for the experiment. Default is 100.
        min_wait_time_in_ns (int): Minimum idle wait time in nanoseconds. Default is 16.
        max_wait_time_in_ns (int): Maximum idle wait time in nanoseconds. Default is 100000.
        wait_time_step_in_ns (int): Step size for idle wait time in nanoseconds. Default is 600.
        reset_type (Literal["active", "thermal"]): Type of qubit reset. Default is "thermal".
        use_state_discrimination (bool): Whether to use state discrimination. Default is False.
    """
    
    num_averages: int = 100
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 100000
    wait_time_step_in_ns: int = 600
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = False
   

def get_idle_times_in_clock_cycles(parameters) -> np.ndarray:
    """
    Get the idle times in clock cycles.
    Args:
        parameters (Parameters): Parameters for the experiment.
    Returns:
        np.ndarray: Array of idle times in clock cycles.
    """
    return np.arange(
        parameters.min_wait_time_in_ns // 4,
        parameters.max_wait_time_in_ns // 4,
        parameters.wait_time_step_in_ns // 4,
    )
    
def get_arb_flux_offset_for_each_qubit(qubits, flux_point) -> dict:
        """
        Get the arbitrary flux bias offset for each qubit.
        Args:
            qubits (List[Qubit]): List of qubits.
            flux_point (str): Type of flux point control.
        Returns:    
            dict: Dictionary of arbitrary flux bias offset for each qubit.
        """
        
        if flux_point == "arbitrary":
            arb_flux_offset = {q.name: q.z.arbitrary_offset for q in qubits}
        else:
            arb_flux_offset = {q.name: 0.0 for q in qubits}

        
        return arb_flux_offset
from typing import List, Literal, Optional

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
        qubits (Optional[List[str]]): List of qubits to be used in the experiment.
        num_averages (int): Number of averages for the experiment. Default is 100.
        min_wait_time_in_ns (int): Minimum idle wait time in nanoseconds. Default is 16.
        max_wait_time_in_ns (int): Maximum idle wait time in nanoseconds. Default is 100000.
        wait_time_step_in_ns (int): Step size for idle wait time in nanoseconds. Default is 600.
        flux_point_joint_or_independent_or_arbitrary (Literal["joint", "independent", "arbitrary"]): 
            Type of flux point control. Default is "independent".
        reset_type (Literal["active", "thermal"]): Type of qubit reset. Default is "thermal".
        use_state_discrimination (bool): Whether to use state discrimination. Default is False.
        simulate (bool): Whether to run the experiment in simulation mode. Default is False.
        simulation_duration_ns (int): Duration of the simulation in nanoseconds. Default is 2500.
        timeout (int): Timeout for the experiment. Default is 100.
        load_data_id (Optional[int]): ID of the data to load. Default is None.
        multiplexed (bool): Whether the experiment is multiplexed. Default is False.
    """
    
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 100000
    wait_time_step_in_ns: int = 600
    flux_point_joint_or_independent_or_arbitrary: Literal["joint", "independent", "arbitrary"] = "independent"
    reset_type: Literal["active", "thermal"] = "thermal"
    use_state_discrimination: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    use_waveform_report: bool = False

from qualibrate import NodeParameters
from typing import List, Optional, Literal

# TODO : documentation
class Parameters(NodeParameters):
    """
    Parameters for the Power Rabi experiment.

    Attributes:
        qubits (Optional[List[str]]): List of qubits to be used in the experiment. If None, the QuAM active qubits are used.
        num_averages (int): Number of averages for the experiment. Default is 50.
        operation_x180_or_any_90 (Literal["x180", "x90", "-x90", "y90", "-y90"]): Type of operation to be performed. Default is "x180".
        min_amp_factor (float): Minimum amplitude factor for the Rabi pulses. Default is 0.001.
        max_amp_factor (float): Maximum amplitude factor for the Rabi pulses. Default is 1.99.
        amp_factor_step (float): Step size for the amplitude factor. Default is 0.005.
        max_number_rabi_pulses_per_sweep (int): Maximum number of Rabi pulses per sweep. Default is 1.
        flux_point_joint_or_independent (Literal["joint", "independent"]): Type of flux offset setting to apply on qubits, either "joint" or "independent".
        reset_type_thermal_or_active (Literal["thermal", "active"]): Type of reset, either "thermal" or "active". Default is "thermal".
        state_discrimination (bool): Whether to use state discrimination. Default is False.
        update_x90 (bool): Whether to update x90 drive amplitude in QuAM object. Default is True.
        simulate (bool): Whether to simulate the experiment. Default is False.
        simulation_duration_ns (int): Duration of the simulation in nanoseconds. Default is 2500.
        timeout (int): time in seconds for qm_session to wait for resources to free up before raising exception. Default is 100.
        load_data_id (Optional[int]): id of the data to be loaded. Default is None.
        multiplexed (bool): Whether the experiment is multiplexed. Default is True.
    """

    qubits: Optional[List[str]] = None
    num_averages: int = 50
    operation_x180_or_any_90: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    min_amp_factor: float = 0.001
    max_amp_factor: float = 1.99
    amp_factor_step: float = 0.005
    max_number_rabi_pulses_per_sweep: int = 1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    state_discrimination: bool = False
    update_x90: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

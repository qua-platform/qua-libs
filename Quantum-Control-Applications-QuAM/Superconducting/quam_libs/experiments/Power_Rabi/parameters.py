from qualibrate import NodeParameters
from typing import List, Optional, Literal

from quam_libs.experiments.node_parameters import DataLoadableNodeParameters, FluxControlledNodeParameters, MultiplexableNodeParameters, QmSessionNodeParameters, QubitsExperimentNodeParameters, SimulatableNodeParameters

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
    Parameters for the Power Rabi experiment.

    Attributes:
        num_averages (int): Number of averages for the experiment. Default is 50.
        operation_x180_or_any_90 (Literal["x180", "x90", "-x90", "y90", "-y90"]): Type of operation to be performed. Default is "x180".
        min_amp_factor (float): Minimum amplitude factor for the Rabi pulses. Default is 0.001.
        max_amp_factor (float): Maximum amplitude factor for the Rabi pulses. Default is 1.99.
        amp_factor_step (float): Step size for the amplitude factor. Default is 0.005.
        max_number_rabi_pulses_per_sweep (int): Maximum number of Rabi pulses per sweep. Default is 1.
        reset_type_thermal_or_active (Literal["thermal", "active"]): Type of reset, either "thermal" or "active". Default is "thermal".
        state_discrimination (bool): Whether to use state discrimination. Default is False.
        update_x90 (bool): Whether to update x90 drive amplitude in QuAM object. Default is True.
    """

    num_averages: int = 50
    operation_x180_or_any_90: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    min_amp_factor: float = 0.001
    max_amp_factor: float = 1.99
    amp_factor_step: float = 0.005
    max_number_rabi_pulses_per_sweep: int = 1
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    state_discrimination: bool = False
    update_x90: bool = True

import numpy as np
from qualibrate import NodeParameters
from typing import List, Optional, Literal

from quam_libs.experiments.node_parameters import DataLoadableNodeParameters, FluxControlledNodeParameters, MultiplexableNodeParameters, QmSessionNodeParameters, QubitsExperimentNodeParameters, SimulatableNodeParameters

from qualibrate.parameters import RunnableParameters

class RabiParameters(RunnableParameters):
    
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
    

def get_number_of_rabi_pulses(parameters: RabiParameters) -> np.ndarray:
    
    N_pi = parameters.max_number_rabi_pulses_per_sweep
    operation = parameters.operation_x180_or_any_90
    if N_pi > 1:
        if operation == "x180":
            N_rabi_pulses = _get_number_of_pi_pulses_for_x180_rabi(parameters)
        elif operation in ["x90", "-x90", "y90", "-y90"]:
            N_rabi_pulses = _get_number_of_pi_half_pulses_for_x90_rabi(parameters)
        else:
            raise ValueError(f"Unrecognized operation {operation}.")
    else:
        N_rabi_pulses = np.linspace(1, N_pi, N_pi).astype("int")[::2]
        
    return N_rabi_pulses

def _get_number_of_pi_pulses_for_x180_rabi(parameters: RabiParameters):
   """
   Generates the sweep axis for the number of pi-pulses to be applied in a Rabi experiment.

   The axis starts at 1 and steps by 2 each time so that on-resonance, the state-population is always
   high
   """
   
   N_pi = parameters.max_number_rabi_pulses_per_sweep
   
   return np.arange(1, N_pi, 2).astype("int")

def _get_number_of_pi_half_pulses_for_x90_rabi(parameters: RabiParameters):
   """
   Generates the sweep axis for the number of pi/2-pulses to be applied in a Rabi experiment.

   The axis starts at 2 and steps by 4 each time so that on-resonance, the state-population is always
   high
   """
   
   N_pi_half = parameters.max_number_rabi_pulses_per_sweep
   
   return np.arange(2, N_pi_half, 4).astype("int")
    

class Parameters(   
    NodeParameters,
    SimulatableNodeParameters,
    DataLoadableNodeParameters,
    QmSessionNodeParameters,
    RabiParameters,
    FluxControlledNodeParameters,
    MultiplexableNodeParameters,
    QubitsExperimentNodeParameters,
): 
    pass
from quam.core import quam_dataclass
from quam.components.channels import MWChannel
from base_transmon import BaseTransmon
from ..architectural_elements.readout_resonator import ReadoutResonator
from typing import Dict, Any, Union, List, Tuple
from dataclasses import field

__all__ = ["Transmon"]


@quam_dataclass
class Transmon(BaseTransmon):
    """
    Example QuAM component for a transmon qubit.

    Args:
        id (str, int): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        xy (MWChannel): The xy drive component.
        resonator (ReadoutResonator): The readout resonator component.
        T1 (float): The transmon T1 in s.
        T2ramsey (float): The transmon T2* in s.
        T2echo (float): The transmon T2 in s.
        thermalization_time_factor (int): thermalization time in units of T1.
        anharmonicity (int, float): the transmon anharmonicity in Hz.
        sigma_time_factor:
        GEF_frequency_shift (int):
        chi (float):
        grid_location (str): qubit location in the plot grid as "(column, row)"
    """

    id: Union[int, str]

    xy: MWChannel = None
    xy_detuned: MWChannel = None
    resonator: ReadoutResonator = None

    RB_fidelity: float = None

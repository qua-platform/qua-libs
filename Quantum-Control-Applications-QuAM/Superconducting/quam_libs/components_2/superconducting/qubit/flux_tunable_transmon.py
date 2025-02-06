from quam.core import quam_dataclass
from quam_libs.components_2.superconducting.qubit.fixed_frequency_transmon import FixedFrequencyTransmon
from quam_libs.components_2.superconducting.architectural_elements.flux_line import FluxLine
from typing import Union

__all__ = ["FluxTunableTransmon"]


@quam_dataclass
class FluxTunableTransmon(FixedFrequencyTransmon):
    """
    Example QuAM component for a transmon qubit.

    Args:
        id (str, int): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        z (FluxLine): The z drive component.
        resonator (ReadoutResonator): The readout resonator component.
        freq_vs_flux_01_quad_term (float):
        arbitrary_intermediate_frequency (float):
        phi0_current (float):
        phi0_voltage (float):
    """

    id: Union[int, str]

    z: FluxLine = None
    freq_vs_flux_01_quad_term: float = 0.0
    arbitrary_intermediate_frequency: float = 0.0
    phi0_current: float = 0.0
    phi0_voltage: float = 0.0

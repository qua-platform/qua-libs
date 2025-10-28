from quam.core import quam_dataclass
from quam.components.channels import IQChannel
from quam import QuamComponent
from typing import Union
from qm.qua import align, wait
import numpy as np

__all__ = ["TWPA"]


@quam_dataclass
class TWPA(QuamComponent):
    """
    Example QuAM component for a TWPA.

    Args:
        id (str, int): The id of the TWPA, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        pump (IQChannel): The pump component
        
    """

    id: Union[int, str]

    pump: IQChannel = None
    spectroscopy: IQChannel = None

    max_gain: float = None
    max_snr_improvement: float = None
    p_saturation: float = None

    pump_frequency : float = None
    pump_amplitude : float = None


    dispersive_feature: float = None
    qubits: list = None

  
    def get_output_power(self, operation, Z=50) -> float:
        power = self.xy.opx_output.full_scale_power_dbm
        amplitude = self.xy.operations[operation].amplitude
        x_mw = 10 ** (power / 10)                       #Pmw
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000) # Vp
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z) # Pdbm

    
    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

   
    def align(self, other = None):
        channels = [self.xy.name, self.resonator.name, self.z.name]

        if other is not None:
            channels += [other.xy.name, other.resonator.name, other.z.name]

        align(*channels)

    def wait(self, duration):
        wait(duration, self.xy.name, self.z.name, self.resonator.name)

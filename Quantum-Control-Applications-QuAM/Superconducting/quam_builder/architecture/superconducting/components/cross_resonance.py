from quam.core import quam_dataclass
from quam.components.channels import IQChannel, MWChannel

__all__ = ["CrossResonanceIQ", "CrossResonanceMW"]


@quam_dataclass
class CrossResonanceBase:
    """
    Example QuAM component for a cross resonance gate.

    Attributes:
        target_qubit_LO_frequency (float): the coupler flux bias for which the interaction is off.
        target_qubit_IF_frequency (float): the coupler flux bias for which the interaction is ON.
        bell_state_fidelity (float): an arbitrary coupler flux bias.
    """

    target_qubit_LO_frequency: float = None
    target_qubit_IF_frequency: float = None
    bell_state_fidelity: float = None


@quam_dataclass
class CrossResonanceIQ(IQChannel, CrossResonanceBase):

    @property
    def upconverter_frequency(self):
        return self.LO_frequency

    @property
    def inferred_intermediate_frequency(self):
        return self.target_qubit_LO_frequency + self.target_qubit_IF_frequency - self.LO_frequency


@quam_dataclass
class CrossResonanceMW(MWChannel, CrossResonanceBase):
    @property
    def inferred_intermediate_frequency(self):
        return self.target_qubit_LO_frequency + self.target_qubit_IF_frequency - self.LO_frequency

    @property
    def upconverter_frequency(self):
        return self.opx_output.upconverter_frequency

    @property
    def inferred_RF_frequency(self):
        return self.upconverter_frequency + self.inferred_intermediate_frequency

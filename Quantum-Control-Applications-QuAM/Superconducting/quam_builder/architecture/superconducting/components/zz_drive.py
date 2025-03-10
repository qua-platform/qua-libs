from quam.core import quam_dataclass
from quam.components.channels import IQChannel, MWChannel

__all__ = ["ZZDriveIQ", "ZZDriveMW"]


@quam_dataclass
class ZZDriveBase:
    target_qubit_LO_frequency: int = None
    target_qubit_IF_frequency: int = None
    detuning: int = None


@quam_dataclass
class ZZDriveIQ(IQChannel, ZZDriveBase):

    @property
    def upconverter_frequency(self):
        return self.LO_frequency

    @property
    def inferred_intermediate_frequency(self):
        return (
            self.target_qubit_LO_frequency
            + self.target_qubit_IF_frequency
            - self.LO_frequency
            + self.detuning
        )


@quam_dataclass
class ZZDriveMW(MWChannel, ZZDriveBase):
    @property
    def inferred_intermediate_frequency(self):
        return (
            self.target_qubit_LO_frequency
            + self.target_qubit_IF_frequency
            - self.LO_frequency
            + self.detuning
        )

    @property
    def upconverter_frequency(self):
        return self.opx_output.upconverter_frequency

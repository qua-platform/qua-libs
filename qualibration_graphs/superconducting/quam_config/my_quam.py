from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam
from qm import QuantumMachinesManager
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam

from .cloud_infrastructure import CloudQuantumMachinesManager


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(FluxTunableQuam):

    def connect(self) -> QuantumMachinesManager:
        """Open a Quantum Machine Manager with the credentials ("host" and "cluster_name") as defined in the network file.

        Returns: the opened Quantum Machine Manager.
        """
        if self.network.get("cloud", False):
            self.qmm = CloudQuantumMachinesManager(self.network["quantum_computer_backend"])
        else:
            settings = dict(
                host=self.network["host"],
                cluster_name=self.network["cluster_name"],
                octave=self.get_octave_config(),
            )

            if "port" in self.network:
                settings["port"] = self.network["port"]

            self.qmm = QuantumMachinesManager(**settings)

        return self.qmm

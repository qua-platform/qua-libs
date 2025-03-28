# from quam_builder.architecture.superconducting.qpu import BaseQuAM
# from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuAM
from typing import ClassVar, Optional

from qm import QuantumMachinesManager
from quam_builder.architecture.superconducting.qpu import FluxTunableQuAM

from .cloud_infrastructure import CloudQuantumMachinesManager

BaseQuAM = FluxTunableQuAM
# BaseQuAM = FixedFrequencyQuAM

# BaseQuAM = BaseQuAM  # use this for a clean-slate, custom QuAM


class QuAM(BaseQuAM):

    qmm: ClassVar[Optional[QuantumMachinesManager]] = None

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

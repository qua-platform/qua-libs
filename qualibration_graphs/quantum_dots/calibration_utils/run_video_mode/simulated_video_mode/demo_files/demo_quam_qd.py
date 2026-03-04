from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam.core import quam_dataclass
from qm import QuantumMachinesManager
from qm_wrapper import QMW

@quam_dataclass
class DemoQuam(BaseQuamQD): 
    """
    Quam state specifically to be used for the demo of MM.
    """
    def connect(self) -> QuantumMachinesManager:
        """Open a Quantum Machine Manager with the credentials ("host" and "cluster_name") as defined in the network file.

        Returns:
            QuantumMachinesManager: The opened Quantum Machine Manager.
        """
        settings = dict(
            host=self.network["host"],
            cluster_name=self.network["cluster_name"],
            octave=self.get_octave_config(),
        )
        if "port" in self.network:
            settings["port"] = self.network["port"]
        self.qmm = QMW(**settings)
        return self.qmm

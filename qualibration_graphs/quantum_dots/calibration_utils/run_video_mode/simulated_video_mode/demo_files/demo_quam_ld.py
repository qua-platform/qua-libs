from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam
from quam.core import quam_dataclass
from qm import QuantumMachinesManager
from calibration_utils.run_video_mode.simulated_video_mode.demo_files.qm_wrapper import QMW


@quam_dataclass
class DemoQuamLD(LossDiVincenzoQuam):
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

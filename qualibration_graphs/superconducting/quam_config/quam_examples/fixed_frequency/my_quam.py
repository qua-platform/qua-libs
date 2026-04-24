from qm import QuantumMachinesManager
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FixedFrequencyZZDriveQuam

# # Define the QUAM class that will be used in all calibration nodes
# # Should inherit from either FixedFrequencyQuam or FluxTunableQuam
class Quam(FixedFrequencyZZDriveQuam):
    pass

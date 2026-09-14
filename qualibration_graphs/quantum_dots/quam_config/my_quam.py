from quam.core import quam_dataclass
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD, LossDiVincenzoQuam


# Define the QUAM class that will be used in all calibration nodes
# This inherits right now from LossDiVincenzoQuam, which is the Qubit layer
# If you only need the QD layer, change this line to inherit from BaseQuamQD
@quam_dataclass
class Quam(LossDiVincenzoQuam):
    pass

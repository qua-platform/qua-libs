from quam.core import quam_dataclass
from quam_builder.architecture.quantum_dots.qpu import LossDiVincenzoQuam


# Define the QUAM class that will be used in all calibration nodes
# LossDiVincenzoQuam is able to perform the HW level calibrations, since it inherits from BaseQuamQD
@quam_dataclass
class Quam(LossDiVincenzoQuam):
    pass

from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam, ParametricQuam

from quam.core import quam_dataclass


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(ParametricQuam):
    pass



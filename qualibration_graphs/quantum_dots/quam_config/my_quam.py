
from quam.core import quam_dataclass
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from BaseQuamQD
@quam_dataclass
class Quam(BaseQuamQD): 
    pass


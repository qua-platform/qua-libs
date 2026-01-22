from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import (
    FixedFrequencyQuam,
    FluxTunableQuam,
    FixedFrequencyTransmonSingleCavityQuam,
)


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from one of the following depending on your setup:
#   - FixedFrequencyQuam: For fixed-frequency transmons without cavities
#   - FluxTunableQuam: For flux-tunable transmons without cavities
#   - FixedFrequencyTransmonSingleCavityQuam: For fixed-frequency transmons with bosonic cavity modes
@quam_dataclass
class Quam(FluxTunableQuam):
    pass

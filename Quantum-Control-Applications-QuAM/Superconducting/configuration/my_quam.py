from quam_libs.components.superconducting.qpu.flux_tunable_qpu import QuAM as FluxTunableQuAM
from quam_libs.components.superconducting.qpu.fixed_frequency_qpu import QuAM as FixedFrequencyQuAM
from enum import Enum

class Architecture(Enum):
    FIXED_FREQUENCY = 1
    FLUX_TUNABLE = 2
    CUSTOM = 3

get_quam = {
    Architecture.FLUX_TUNABLE: FluxTunableQuAM(),
    Architecture.FIXED_FREQUENCY: FixedFrequencyQuAM(),
}

QuAM = get_quam[Architecture.FLUX_TUNABLE]

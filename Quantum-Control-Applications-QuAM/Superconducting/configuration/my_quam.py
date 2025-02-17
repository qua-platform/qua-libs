from quam_libs.components.superconducting.qpu.base_quam import BaseQuAM
from quam_libs.components.superconducting.qpu.flux_tunable_qpu import QuAM as FluxTunableQuAM
from quam_libs.components.superconducting.qpu.fixed_frequency_qpu import QuAM as FixedFrequencyQuAM


BaseQuAM = FluxTunableQuAM
# BaseQuAM = FixedFrequencyQuAM

# BaseQuAM = BaseQuAM  # use this for a clean-slate, custom QuAM

class QuAM(BaseQuAM):
    pass

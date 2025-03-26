# from quam_builder.architecture.superconducting.qpu import BaseQuAM
from quam_builder.architecture.superconducting.qpu import FluxTunableQuAM

# from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuAM


BaseQuAM = FluxTunableQuAM
# BaseQuAM = FixedFrequencyQuAM

# BaseQuAM = BaseQuAM  # use this for a clean-slate, custom QuAM


class QuAM(BaseQuAM):
    pass

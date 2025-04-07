# from quam_builder.architecture.superconducting.qpu import BaseQuAM
from quam_builder.architecture.superconducting.qpu import FluxTunableQuAM

# from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuAM


BaseQuam = FluxTunableQuAM
# BaseQuAM = FixedFrequencyQuAM

# BaseQuAM = BaseQuAM  # use this for a clean-slate, custom QuAM


class Quam(BaseQuam):
    pass

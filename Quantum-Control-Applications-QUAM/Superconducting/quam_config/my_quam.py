# from quam_builder.architecture.superconducting.qpu import BaseQuam
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam

# from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam


BaseQuam = FluxTunableQuam
# BaseQuam = FixedFrequencyQuam

# BaseQuam = BaseQuam  # use this for a clean-slate, custom QUAM


class Quam(BaseQuam):
    pass

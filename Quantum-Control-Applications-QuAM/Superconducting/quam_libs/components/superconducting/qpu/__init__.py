from quam_libs.components.superconducting.qpu.base_quam import BaseQuAM
from quam_libs.components.superconducting.qpu.fixed_frequency_qpu import QuAM as FixedFrequencyQuAM
from quam_libs.components.superconducting.qpu.flux_tunable_qpu import QuAM as FluxTunableQuAM
from typing import Union

__all__ = [
    *base_quam.__all__,
    *fixed_frequency_qpu.__all__,
    *flux_tunable_qpu.__all__,
]

AnyQuAM = Union[BaseQuAM, FixedFrequencyQuAM, FluxTunableQuAM]

from quam_builder.architecture.superconducting.qubit_pair.flux_tunable_transmons import (
    FluxTunableTransmonPair,
)
from quam_builder.architecture.superconducting.qubit_pair.fixed_frequency_transmons import (
    FixedFrequencyTransmonPair,
)
from typing import Union

__all__ = [
    *fixed_frequency_transmons.__all__,
    *flux_tunable_transmons.__all__,
]

AnyTransmonPair = Union[FixedFrequencyTransmonPair, FluxTunableTransmonPair]

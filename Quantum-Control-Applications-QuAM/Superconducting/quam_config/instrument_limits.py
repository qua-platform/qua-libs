from dataclasses import dataclass
from typing import Union

from quam.components.channels import IQChannel, MWChannel


@dataclass(frozen=True)
class InstrumentLimits:
    max_wf_amplitude: float
    max_x180_wf_amplitude: float
    max_readout_amplitude: float
    units: str


def instrument_limits(channel: Union[IQChannel, MWChannel]) -> InstrumentLimits:
    # Todo: these parameters should be accessible to the user
    if not (isinstance(channel, IQChannel) ^ isinstance(channel, MWChannel)):
        raise TypeError(f"Expected channel to be type IQChannel xor MWChannel for type checking, got {type(channel)}.")

    if isinstance(channel, MWChannel):
        limits = InstrumentLimits(
            # MW-FEM max normalized amplitude
            max_wf_amplitude=1,
            # A subjective "safe" value for x180 pulses
            max_x180_wf_amplitude=0.6,
            # A subjective "safe" value assuming up to 10 qubits on the same channel
            max_readout_amplitude=0.1,
            units="(scaled by `full_scale_power_dbm`)",
        )
    elif isinstance(channel, IQChannel):
        limits = InstrumentLimits(
            # OPX+ and LF-FEM not in amplified-mode
            max_wf_amplitude=0.5,
            # A subjective "safe" value for x180 pulses
            max_x180_wf_amplitude=0.3,
            # A subjective "safe" value assuming up to 10 qubits on the same channel
            max_readout_amplitude=0.05,
            units="V",
        )
    else:
        raise TypeError()

    return limits

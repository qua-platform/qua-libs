from typing import List, Type

from ..connectivity.wiring_spec_enums import WiringIOType
from ..instruments.instrument_channels import (
    InstrumentChannelMwFemInput,
    InstrumentChannelMwFemOutput,
    InstrumentChannelOctaveInput,
    InstrumentChannelOctaveOutput,
    InstrumentChannelLfFemOutput,
    InstrumentChannelLfFemInput,
    InstrumentChannelOpxPlusInput,
    InstrumentChannelOpxPlusOutput,
    InstrumentChannel,
)


def get_channel_mapping(
    allocation_type: str, io_type: WiringIOType, num: int = 1
) -> List[Type[InstrumentChannel]]:
    return {
        "lf-fem": {
            WiringIOType.INPUT: [InstrumentChannelLfFemInput] * num,
            WiringIOType.OUTPUT: [InstrumentChannelLfFemOutput] * num,
            WiringIOType.INPUT_AND_OUTPUT: [InstrumentChannelLfFemInput] * num
            + [InstrumentChannelLfFemOutput] * num,
        },
        "mw-fem": {
            WiringIOType.INPUT: [InstrumentChannelMwFemInput] * num,
            WiringIOType.OUTPUT: [InstrumentChannelMwFemOutput] * num,
            WiringIOType.INPUT_AND_OUTPUT: [InstrumentChannelMwFemInput] * num
            + [InstrumentChannelMwFemOutput],
        },
        "octave": {
            WiringIOType.INPUT: [InstrumentChannelOctaveInput] * num,
            WiringIOType.OUTPUT: [InstrumentChannelOctaveOutput] * num,
            WiringIOType.INPUT_AND_OUTPUT: [InstrumentChannelOctaveInput] * num
            + [InstrumentChannelOctaveOutput] * num,
        },
        "opx+": {
            WiringIOType.INPUT: [InstrumentChannelOpxPlusInput] * num,
            WiringIOType.OUTPUT: [InstrumentChannelOpxPlusOutput] * num,
            WiringIOType.INPUT_AND_OUTPUT: [InstrumentChannelOpxPlusInput] * num
            + [InstrumentChannelOpxPlusOutput] * num,
        },
    }[allocation_type][io_type]

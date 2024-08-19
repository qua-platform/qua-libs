from typing import List, Union
from .instrument_channels import *
from .constants import *


class Instruments:
    """
    Class to add the static information about which QM instruments will be used
    in an experimental setup. Upon adding an instrument, its available channels
    will be enumerated and added individually to a stack of free channels.
    """

    def __init__(self):
        self.available_channels = InstrumentChannels()

    def add_octave(self, indices: Union[List[int], int]):
        if isinstance(indices, int):
            indices = [indices]

        for index in indices:
            for port in range(1, NUM_OCTAVE_INPUT_PORTS + 1):
                channel = InstrumentChannelOctaveInput(con=index, port=port)
                self.available_channels.add(channel)

            for port in range(1, NUM_OCTAVE_OUTPUT_PORTS + 1):
                channel = InstrumentChannelOctaveOutput(con=index, port=port)
                self.available_channels.add(channel)

    def add_lf_fem(self, con: int, slots: Union[List[int], int]):
        if isinstance(slots, int):
            slots = [slots]

        for slot in slots:
            for port in range(1, NUM_LF_FEM_INPUT_PORTS + 1):
                channel = InstrumentChannelLfFemInput(con=con, slot=slot, port=port)
                self.available_channels.add(channel)

            for port in range(1, NUM_LF_FEM_OUTPUT_PORTS + 1):
                channel = InstrumentChannelLfFemOutput(con=con, slot=slot, port=port)
                self.available_channels.add(channel)

    def add_mw_fem(self, con: int, slots: Union[List[int], int]):
        if isinstance(slots, int):
            slots = [slots]

        for slot in slots:
            for port in range(1, NUM_MW_FEM_INPUT_PORTS + 1):
                channel = InstrumentChannelMwFemInput(con=con, slot=slot, port=port)
                self.available_channels.add(channel)

            for port in range(1, NUM_MW_FEM_OUTPUT_PORTS + 1):
                channel = InstrumentChannelMwFemOutput(con=con, slot=slot, port=port)
                self.available_channels.add(channel)

    def add_opx_plus(self, cons: Union[List[int], int]):
        if isinstance(cons, int):
            cons = [cons]

        for con in cons:
            for port in range(1, NUM_OPX_PLUS_INPUT_PORTS + 1):
                channel = InstrumentChannelOpxPlusInput(con=con, port=port)
                self.available_channels.add(channel)

            for port in range(1, NUM_OPX_PLUS_OUTPUT_PORTS + 1):
                channel = InstrumentChannelOpxPlusOutput(con=con, port=port)
                self.available_channels.add(channel)

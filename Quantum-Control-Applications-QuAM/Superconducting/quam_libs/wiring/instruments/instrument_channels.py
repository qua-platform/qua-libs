from dataclasses import dataclass
from typing import Union, Literal, Type


@dataclass
class InstrumentChannel:
    con: int
    port: int
    slot: Union[None, int] = None
    io_type: Literal["input", "output"] = None

    def __str__(self):
        return f'({", ".join([f"con{self.con}", f"{self.slot}" if self.slot else "", str(self.port)])})'


@dataclass
class InstrumentChannelInput(InstrumentChannel):
    io_type: Literal["input", "output"] = "input"


@dataclass
class InstrumentChannelOutput(InstrumentChannel):
    io_type: Literal["input", "output"] = "output"


class InstrumentChannelLfFemInput(InstrumentChannelInput):
    pass


class InstrumentChannelLfFemOutput(InstrumentChannelOutput):
    pass


class InstrumentChannelMwFemInput(InstrumentChannelInput):
    pass


class InstrumentChannelMwFemOutput(InstrumentChannelOutput):
    pass


class InstrumentChannelOpxPlusInput(InstrumentChannelInput):
    pass


class InstrumentChannelOpxPlusOutput(InstrumentChannelOutput):
    pass


class InstrumentChannelOctaveInput(InstrumentChannelInput):
    pass


class InstrumentChannelOctaveOutput(InstrumentChannelOutput):
    pass


CHANNELS_OPX_PLUS = [InstrumentChannelOpxPlusInput, InstrumentChannelOpxPlusOutput]

CHANNELS_OPX_1000 = [
    InstrumentChannelLfFemInput,
    InstrumentChannelLfFemOutput,
    InstrumentChannelMwFemInput,
    InstrumentChannelMwFemOutput,
]


class InstrumentChannels:
    """
    Collection of "stack" data-structures organized by channel type.
    Each entry contains a list of instrument channels. This wrapper
    can be interacted with as if it were the underlying stack dictionary.
    """

    def __init__(self):
        self.stack = {}

    def check_if_already_occupied(self, channel: InstrumentChannel):
        for channel_type in self.stack:
            for existing_channel in self.stack[channel_type]:
                if (
                    channel.con == existing_channel.con
                    and channel.slot == existing_channel.slot
                    and channel.port == existing_channel.port
                    and channel.io_type == existing_channel.io_type
                ):
                    if channel.slot is None:
                        if type(channel) != type(existing_channel):
                            pass
                        else:
                            raise ValueError(
                                f"{channel.io_type} channel on con{channel.con}, "
                                f"port {channel.port} is already occupied."
                            )
                    else:
                        raise ValueError(
                            f"{channel.io_type} channel on con{channel.con}, "
                            f"slot {channel.slot}, port {channel.port} is already occupied."
                        )

    def check_if_mixing_opx_1000_and_opx_plus(self, channel: InstrumentChannel):
        if type(channel) in CHANNELS_OPX_1000:
            for channel_type in CHANNELS_OPX_PLUS:
                if channel_type in self.stack:
                    raise ValueError(f"Can't add an FEM to a setup with an OPX+.")
        elif type(channel) in CHANNELS_OPX_PLUS:
            for channel_type in CHANNELS_OPX_1000:
                if channel_type in self.stack:
                    raise ValueError(
                        f"Can't add an OPX+ to a setup with an OPX1000 FEM."
                    )

    def add(self, channel: InstrumentChannel):
        self.check_if_already_occupied(channel)
        self.check_if_mixing_opx_1000_and_opx_plus(channel)

        channel_type = type(channel)
        if channel_type not in self.stack:
            self.stack[channel_type] = []

        self.stack[channel_type].append(channel)

    def pop(self, channel: Type[InstrumentChannel]):
        return self.stack[channel].pop(0)

    def get(self, key: InstrumentChannel, fallback=None):
        return self.stack.get(key, fallback)

    def __getitem__(self, item):
        return self.stack[item]

    def __iter__(self):
        return iter(self.stack)

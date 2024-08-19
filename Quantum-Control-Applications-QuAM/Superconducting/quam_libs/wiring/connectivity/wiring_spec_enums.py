from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable


class WiringFrequency(Enum):
    DC = "DC"
    RF = "RF"


class WiringIOType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INPUT_AND_OUTPUT = "input_output"


class WiringLineType(Enum):
    RESONATOR = "rr"
    DRIVE = "xy"
    FLUX = "z"
    COUPLER = "c"


@dataclass
class WiringIOSpec:
    """
    A technical specification for what IO is required in a WiringSpecifcation.
    The wiring type indicates whether the required channels are input, output
    or both. The con/slot/port attributes allow the hardcoding of more
    specific channels to satisfy the wiring.
    """

    type: WiringIOType
    con: Optional[int] = None
    slot: Optional[int] = None
    port: Optional[int] = None

    def make_channel_filter(self) -> Callable[["InstrumentChannel"], bool]:
        # todo:
        return lambda channel: (
            (self.con is None or self.con == channel.con)
            and (self.slot is None or self.slot == channel.slot)
            and (self.port is None or self.port == channel.port)
        )

    def __str__(self):
        return (
            f"IO type={self.type.value}" + f", con=con{self.con}"
            if self.con is not None
            else "" + f", slot={self.slot}"
            if self.slot is not None
            else "" + f", port={self.port}"
            if self.port is not None
            else ""
        )

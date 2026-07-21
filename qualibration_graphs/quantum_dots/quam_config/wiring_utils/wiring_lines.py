from enum import Enum
from typing import Union, List

from typing import TYPE_CHECKING
from qualang_tools.wirer.connectivity.wiring_spec import WiringFrequency, WiringIOType

if TYPE_CHECKING:
    from qualang_tools.wirer.connectivity.channel_spec import ChannelSpec
    from qualang_tools.wirer.connectivity.element import Element


class WiringLineType(Enum):
    RESONATOR = "rr"
    DRIVE = "xy"
    PLUNGER = "p"
    BARRIER = "b"
    SENSOR = "s"


RESONATOR = WiringLineType.RESONATOR
DRIVE = WiringLineType.DRIVE
PLUNGER = WiringLineType.PLUNGER
BARRIER = WiringLineType.BARRIER
SENSOR = WiringLineType.SENSOR


class WiringSpec:
    """
    A technical specification for the wiring that will be required to
    manipulate the given quantum elements.
    """

    def __init__(
        self,
        frequency: WiringFrequency,
        io_type: WiringIOType,
        line_type: WiringLineType,
        triggered: bool,
        constraints: "ChannelSpec",
        elements: Union["Element", List["Element"]],
    ):
        self.frequency = frequency
        self.io_type = io_type
        self.line_type = line_type
        self.triggered = triggered
        self.constraints = constraints
        if not isinstance(elements, list):
            elements = [elements]
        self.elements: List["Element"] = elements

from typing import Union, List

from .element import Element
from .wiring_spec_enums import *


class WiringSpec:
    """
    A technical specification for the wiring that will be required to
    manipulate the given quantum elements.
    """

    def __init__(
        self,
        frequency: WiringFrequency,
        io_spec: WiringIOSpec,
        line_type: WiringLineType,
        elements: Union[Element, List[Element]],
    ):
        self.frequency = frequency
        self.io_spec = io_spec
        self.line_type = line_type
        if not isinstance(elements, list):
            elements = [elements]
        self.elements: List[Element] = elements

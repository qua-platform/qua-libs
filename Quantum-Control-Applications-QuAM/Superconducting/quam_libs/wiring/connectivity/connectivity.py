import copy
from typing import Dict, List

from .element import Element, ElementId, QubitReference, QubitPairReference
from .types import QubitsType, QubitPairsType
from .wiring_spec import WiringSpec
from .wiring_spec_enums import (
    WiringFrequency,
    WiringIOType,
    WiringLineType,
    WiringIOSpec,
)


class Connectivity:
    """
    This class stores placeholders for the quantum elements which will be used
    in a setup, as well as the wiring specification for each of those elements.
    This includes at what frequency the element will be driven, whether it
    requires input/output or both lines, if it is required on a particular
    module or FEM slot, and what high-level component will be manipulated.
    """

    def __init__(self):
        self.elements: Dict[ElementId, Element] = {}
        self.specs: List[WiringSpec] = []

    def add_resonator_line(
        self, qubits: QubitsType, con: int = None, slot: int = None, port: int = None
    ):
        elements = self._make_qubit_elements(qubits)
        io_spec = WiringIOSpec(
            type=WiringIOType.INPUT_AND_OUTPUT, con=con, slot=slot, port=port
        )
        return self.add_wiring_spec(
            elements,
            WiringFrequency.RF,
            io_spec,
            WiringLineType.RESONATOR,
            shared_line=True,
        )

    def add_qubit_drive_lines(
        self, qubits: QubitsType, con: int = None, slot: int = None, port: int = None
    ):
        elements = self._make_qubit_elements(qubits)
        io_spec = WiringIOSpec(type=WiringIOType.OUTPUT, con=con, slot=slot, port=port)
        return self.add_wiring_spec(
            elements, WiringFrequency.RF, io_spec, WiringLineType.DRIVE
        )

    def add_qubit_flux_lines(
        self, qubits: QubitsType, con: int = None, slot: int = None, port: int = None
    ):
        elements = self._make_qubit_elements(qubits)
        io_spec = WiringIOSpec(type=WiringIOType.OUTPUT, con=con, slot=slot, port=port)
        return self.add_wiring_spec(
            elements, WiringFrequency.DC, io_spec, WiringLineType.FLUX
        )

    def add_qubit_pair_flux_lines(
        self,
        qubit_pairs: QubitPairsType,
        con: int = None,
        slot: int = None,
        port: int = None,
    ):
        elements = self._make_qubit_pair_elements(qubit_pairs)
        io_spec = WiringIOSpec(type=WiringIOType.OUTPUT, con=con, slot=slot, port=port)
        return self.add_wiring_spec(
            elements, WiringFrequency.DC, io_spec, WiringLineType.COUPLER
        )

    def add_wiring_spec(
        self,
        elements: List[Element],
        frequency: WiringFrequency,
        io_spec: WiringIOSpec,
        line_type: WiringLineType,
        shared_line: bool = False,
    ):
        specs = []
        if shared_line:
            spec = WiringSpec(frequency, io_spec, line_type, elements)
            specs.append(spec)
        else:
            for element in elements:
                io_spec = copy.deepcopy(io_spec)
                spec = WiringSpec(frequency, io_spec, line_type, element)
                specs.append(spec)
        self.specs.extend(specs)

        return specs

    def _make_qubit_elements(self, qubits: QubitsType):
        if not isinstance(qubits, list):
            qubits = [qubits]

        elements = []
        for qubit in qubits:
            id = QubitReference(qubit)
            if id not in self.elements:
                self.elements[id] = Element(id)
            elements.append(self.elements[id])

        return elements

    def _make_qubit_pair_elements(self, qubit_pairs: QubitPairsType):
        if not isinstance(qubit_pairs, list):
            qubit_pairs = [qubit_pairs]

        elements = []
        for qubit_pair in qubit_pairs:
            id = QubitPairReference(*qubit_pair)
            if id not in self.elements:
                self.elements[id] = Element(id)
            elements.append(self.elements[id])

        return elements

from qualang_tools.wirer.connectivity.channel_spec import ChannelSpec
from qualang_tools.wirer.connectivity.types import QubitsType, QubitPairsType
from qualang_tools.wirer.connectivity.wiring_spec import WiringFrequency, WiringIOType, WiringLineType
from qualang_tools.wirer.connectivity.connectivity_base import ConnectivityBase


class Connectivity(ConnectivityBase):
    """
    Represents the high-level wiring configuration for an NV center-based QPU setup.

    This class defines and stores placeholders for quantum elements (e.g., qubits and lasers)
    and specifies the wiring requirements for each of their control and readout lines. It enables
    the configuration of line types (e.g., drive, laser, readout), their I/O roles, and associated
    frequency domains (RF or DC), as well as constraints for channel allocation.

    The API is designed to model single NV centers, along with pairwise coupling mechanisms.
    """

    def add_nv_center(self, qubits: QubitsType):
        """
        Adds specifications (placeholders) for resonator and drive lines for nv centers.

        This method configures the wiring specifications (placeholders) for a set of nv centers,
        including the qubit drive and laser. No channels are allocated at this stage.

        Args:
            qubits (QubitsType): The qubits to configure for nv centers.
        """
        # self.add_laser(qubits)
        self.add_readout(qubits)
        self.add_qubit_drive(qubits)

    def add_laser(self, qubits: QubitsType, triggered: bool = False, constraints: ChannelSpec = None):
        """
        Adds a specification (placeholder) for a laser for the specified qubits.

        This method configures a laser specification (placeholder) that can handle output,
        typically for reading out and initializing the state of qubits. It also allows optional
        triggering and constraints on which channel configurations can be allocated.

        No channels are allocated at this stage.

        Args:
            qubits (QubitsType): The qubits to associate with the laser.
            triggered (bool, optional): Whether the laser is triggered. Defaults to False.
            constraints (ChannelSpec, optional): Constraints on the channel, if any. Defaults to None.

        Returns:
            A wiring specification (placeholder) for the laser.
        """
        elements = self._make_qubit_elements(qubits)
        return self.add_wiring_spec(
            WiringFrequency.DC,
            WiringIOType.OUTPUT,
            "la",  # WiringLineType.RESONATOR,
            triggered,
            constraints,
            elements,
            shared_line=True,
        )

    def add_readout(self, qubits: QubitsType, triggered: bool = False, constraints: ChannelSpec = None):
        """
        Adds a specification (placeholder) for a readout for the specified qubits.

        This method configures a readout specification (placeholder) that can handle input,
        typically for reading out the state of qubits. It also allows optional
        triggering and constraints on which channel configurations can be allocated.

        No channels are allocated at this stage.

        Args:
            qubits (QubitsType): The qubits to associate with the laser.
            triggered (bool, optional): Whether the readout is triggered. Defaults to False.
            constraints (ChannelSpec, optional): Constraints on the channel, if any. Defaults to None.

        Returns:
            A wiring specification (placeholder) for the laser.
        """
        elements = self._make_qubit_elements(qubits)
        return self.add_wiring_spec(
            WiringFrequency.DC,
            WiringIOType.INPUT_AND_OUTPUT,
            "ro",  # WiringLineType.RESONATOR,
            triggered,
            constraints,
            elements,
            shared_line=True,
        )

    def add_qubit_drive(self, qubits: QubitsType, triggered: bool = False, constraints: ChannelSpec = None):
        """
        Adds specifications (placeholders) for drive lines for the specified qubits.

        This method configures the qubit drive line specifications (placeholders), which are typically used to apply
        control signals to qubits. It allows optional triggering and constraints on which channel configurations
        can be allocated for this line.

        No channels are allocated at this stage.


        Args:
            qubits (QubitsType): The qubits to configure the drive lines for.
            triggered (bool, optional): Whether the line is triggered. Defaults to False.
            constraints (ChannelSpec, optional): Constraints on the channel, if any. Defaults to None.

        Returns:
            A wiring specification (placeholder) for the qubit drive lines.
        """
        elements = self._make_qubit_elements(qubits)
        return self.add_wiring_spec(
            WiringFrequency.RF,
            WiringIOType.OUTPUT,
            WiringLineType.DRIVE,
            triggered,
            constraints,
            elements,
            shared_line=True,
        )

    def add_qubit_pair_cross_resonance_lines(
        self, qubit_pairs: QubitPairsType, triggered: bool = False, constraints: ChannelSpec = None
    ):
        """
        Adds specifications (placeholders) for cross-resonance drive lines for a pair of qubits.

        This method configures cross-resonance line specifications (placeholders) for two qubits,
        typically used to implement two-qubit gate operations. One can also specify constraints on which
        channel configurations can be allocated for this line.

        No channels are allocated at this stage.

        Args:
            qubit_pairs (QubitPairsType): The qubit pairs to configure the cross-resonance lines for.
            triggered (bool, optional): Whether the line is triggered. Defaults to False.
            constraints (ChannelSpec, optional): Constraints on the channel, if any. Defaults to None.

        Returns:
            A wiring specification (placeholder) for the cross-resonance drive lines.
        """
        # elements = self._make_qubit_pair_elements(qubit_pairs)
        # return self.add_wiring_spec(
        #     WiringFrequency.RF, WiringIOType.OUTPUT, WiringLineType.CROSS_RESONANCE, triggered, constraints, elements
        # )
        pass

    def add_qubit_pair_zz_drive_lines(
        self, qubit_pairs: QubitPairsType, triggered: bool = False, constraints: ChannelSpec = None
    ):
        """
        Adds specifications (placeholders) for ZZ drive lines for a pair of qubits.

        This method configures ZZ drive line specifications (placeholders) for two qubits, typically used
        for two-qubit gate operations, in the RF frequency domain. One can also specify constraints on which
        channel configurations can be allocated for this line.

        No channels are allocated at this stage.

        Args:
            qubit_pairs (QubitPairsType): The qubit pairs to configure the ZZ drive lines for.
            triggered (bool, optional): Whether the line is triggered. Defaults to False.
            constraints (ChannelSpec, optional): Constraints on the channel, if any. Defaults to None.

        Returns:
            A wiring specification (placeholder) for the ZZ drive lines.
        """
        # elements = self._make_qubit_pair_elements(qubit_pairs)
        # return self.add_wiring_spec(
        #     WiringFrequency.RF, WiringIOType.OUTPUT, WiringLineType.ZZ_DRIVE, triggered, constraints, elements
        # )
        pass

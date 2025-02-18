from typing import Union
from qualang_tools.wirer import Connectivity
from quam_builder.architecture.superconducting.qpu import BaseQuAM, FixedFrequencyQuAM, FluxTunableQuAM
from quam_builder.builder.qop_connectivity.create_wiring import create_wiring
from quam.components.ports import (
    FEMPortsContainer,
    OPXPlusPortsContainer,
)


QuAMType = Union[BaseQuAM, FixedFrequencyQuAM, FluxTunableQuAM]

def build_quam_wiring(
    connectivity: Connectivity, host_ip: str, cluster_name: str,
    quam_instance: QuAMType, port: int = None
):

    machine = quam_instance
    add_ports_container(connectivity, machine)
    add_name_and_ip(machine, host_ip, cluster_name, port)
    machine.wiring = create_wiring(connectivity)
    save_machine(machine)


def add_ports_container(connectivity: Connectivity, machine: QuAMType):
    """
    Detects whether the `connectivity` is using OPX+ or OPX1000 and returns
    the corresponding base object. Otherwise, raises a TypeError.
    """
    for element in connectivity.elements.values():
        for channels in element.channels.values():
            for channel in channels:
                if channel.instrument_id in ["lf-fem", "mw-fem"]:
                    machine.ports = FEMPortsContainer()
                elif channel.instrument_id in ["opx+"]:
                    machine.ports = OPXPlusPortsContainer()



def add_name_and_ip(machine: QuAMType, host_ip: str, cluster_name: str, port: Union[int, None]):
    """Stores the minimal information to connect to a QuantumMachinesManager."""
    machine.network = {"host": host_ip, "port": port, "cluster_name": cluster_name}


def save_machine(machine: QuAMType):
    machine.save(
        content_mapping={
            "wiring.json": ["network", "wiring"],
        },
    )

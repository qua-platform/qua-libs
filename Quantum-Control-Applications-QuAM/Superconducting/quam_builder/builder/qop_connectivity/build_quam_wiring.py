from typing import Union
from qualang_tools.wirer import Connectivity
from quam_builder.architecture.superconducting.qpu import AnyQuAM
from quam_builder.builder.qop_connectivity.create_wiring import create_wiring
from quam.components.ports import FEMPortsContainer, OPXPlusPortsContainer


def build_quam_wiring(
    connectivity: Connectivity,
    host_ip: str,
    cluster_name: str,
    quam_instance: AnyQuAM,
    port: int = None,
):
    """
    Builds the QuAM wiring configuration and saves the machine setup.

    Parameters:
    connectivity (Connectivity): The connectivity configuration.
    host_ip (str): The IP address of the Quantum Orchestration Platform.
    cluster_name (str): The name of the cluster as displayed in the admin panel.
    quam_instance (AnyQuAM): The QuAM instance to be configured.
    port (int, optional): The port number. Defaults to None.
    """
    machine = quam_instance
    add_ports_container(connectivity, machine)
    add_name_and_ip(machine, host_ip, cluster_name, port)
    machine.wiring = create_wiring(connectivity)
    save_machine(machine)


def add_ports_container(connectivity: Connectivity, machine: AnyQuAM):
    """
    Detects whether the `connectivity` is using OPX+ or OPX1000 and returns the corresponding base object.

    Parameters:
    connectivity (Connectivity): The connectivity configuration.
    machine (AnyQuAM): The QuAM machine to which the ports container will be added.

    Raises:
    TypeError: If the instrument type is unknown.
    """
    for element in connectivity.elements.values():
        for channels in element.channels.values():
            for channel in channels:
                if channel.instrument_id in ["lf-fem", "mw-fem"]:
                    machine.ports = FEMPortsContainer()
                elif channel.instrument_id in ["opx+"]:
                    machine.ports = OPXPlusPortsContainer()


def add_name_and_ip(machine: AnyQuAM, host_ip: str, cluster_name: str, port: Union[int, None]):
    """
    Stores the minimal information to connect to a QuantumMachinesManager.

    Parameters:
    machine (AnyQuAM): The QuAM machine to which the network information will be added.
    host_ip (str): The IP address of the host.
    cluster_name (str): The name of the cluster.
    port (Union[int, None]): The port number.
    """
    machine.network = {"host": host_ip, "port": port, "cluster_name": cluster_name}


def save_machine(machine: AnyQuAM):
    """
    Saves the machine configuration to a .json file.

    Parameters:
    machine (AnyQuAM): The QuAM machine to be saved.
    """
    machine.save(
        content_mapping={
            "wiring.json": ["network", "wiring"],
        },
    )

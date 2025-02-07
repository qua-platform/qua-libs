import os
from pathlib import Path
from typing import Union
from qualang_tools.wirer import Connectivity
from quam_libs.components.superconducting.qpu import BaseQuAM, FixedFrequencyQuAM, FluxTunableQuAM
from quam_libs.quam_builder.qop_connectivity.create_wiring import create_wiring
from quam.components.ports import (
    FEMPortsContainer,
    OPXPlusPortsContainer,
)

def build_quam_wiring(
    connectivity: Connectivity, host_ip: str, cluster_name: str, quam_state_path: Union[Path, str], quam_instance: Union[BaseQuAM, FixedFrequencyQuAM, FluxTunableQuAM], port: int = None
):
    # TODO: I removed this because it was raising QUAM warnings, maybe we should consider adding an update_quam_wiring function instead
    # if os.path.exists(quam_state_path) and "state.json" in os.listdir(quam_state_path):
    #     # if there is a non-empty QuAM state already
    #     machine = quam_instance.load(quam_state_path)
    # else:
    machine = quam_instance
    add_ports_container(connectivity, machine)
    add_name_and_ip(machine, host_ip, cluster_name, port)
    machine.wiring = create_wiring(connectivity)
    save_machine(machine, quam_state_path)


def add_ports_container(connectivity: Connectivity, machine: BaseQuAM):
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



def add_name_and_ip(machine, host_ip: str, cluster_name: str, port: Union[int, None]):
    """Stores the minimal information to connect to a QuantumMachinesManager."""
    machine.network = {"host": host_ip, "port": port, "cluster_name": cluster_name}


def save_machine(machine, quam_state_path: Union[Path, str]):
    machine.save(
        path=quam_state_path,
        content_mapping={
            "wiring.json": ["network", "wiring"],
        },
    )

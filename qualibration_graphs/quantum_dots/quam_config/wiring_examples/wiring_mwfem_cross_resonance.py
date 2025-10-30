import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "127.0.0.1"  # QOP IP address
port = None  # QOP Port
cluster_name = "Cluster_1"  # Name of the cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[2, 3, 5, 7])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [i + 1 for i in range(8)]
qubit_pairs = [(i + 1, i + 2) for i in range(7)]
qubit_pairs.append((8, 1))

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
rr_slots = [2, 2, 3, 3, 5, 5, 7, 7]
rr_out_ports = [1, 8, 1, 8, 1, 8, 1, 8]
rr_in_ports = [1, 2, 1, 2, 1, 2, 1, 2]
xy_slots = rr_slots
xy_ports = [2, 3, 2, 3, 2, 3, 2, 3]

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# Single qubit individual drive and readout lines
for i in range(8):
    connectivity.add_resonator_line(
        qubits=qubits[i],
        constraints=mw_fem_spec(con=1, slot=rr_slots[i], in_port=rr_in_ports[i], out_port=rr_out_ports[i]),
    )
    connectivity.add_qubit_drive_lines(
        qubits=qubits[i],
        constraints=mw_fem_spec(con=1, slot=xy_slots[i], out_port=xy_ports[i]),
    )
# Don't block the xy channels to connect the CR and ZZ drives to the same ports
allocate_wiring(connectivity, instruments, block_used_channels=False)
# Two-qubit drives
for i in range(len(qubit_pairs)):
    # Add CR lines
    connectivity.add_qubit_pair_cross_resonance_lines(
        qubit_pairs=qubit_pairs[i],
        constraints=mw_fem_spec(con=1, slot=xy_slots[i], out_port=xy_ports[i]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)
    # Add ZZ lines
    connectivity.add_qubit_pair_zz_drive_lines(
        qubit_pairs=qubit_pairs[i],
        constraints=mw_fem_spec(con=1, slot=xy_slots[i], out_port=xy_ports[i]),
    )
    allocate_wiring(connectivity, instruments, block_used_channels=False)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=True)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
user_input = input("Do you want to save the updated QUAM? (y/n)")
if user_input.lower() == "y":
    machine = Quam()
    # Build the wiring (wiring.json) and initiate the QUAM
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)

    # Reload QUAM, build the QUAM object and save the state as state.json
    machine = Quam.load()
    build_quam(machine)

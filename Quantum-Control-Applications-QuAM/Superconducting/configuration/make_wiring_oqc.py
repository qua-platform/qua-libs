# %%
from pathlib import Path

from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_libs.quam_builder.qop_connectivity.build_quam_wiring import build_quam_wiring
import matplotlib.pyplot as plt
from get_quam import QuAM

# Define static parameters
host_ip = "172.16.33.101"
cluster_name = "Cluster_81"
path = "./quam_state"

# Define the available instrument setup
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[2, 3, 5, 7])

# Define which quantum elements are present in the system
qubits = [i+1 for i in range(8)]
rr_slots = [2, 2, 3, 3, 5, 5, 7, 7]
rr_out_ports = [1, 8, 1, 8, 1, 8, 1, 8]
rr_in_ports = [1, 2, 1, 2, 1, 2, 1, 2]
xy_slots = rr_slots
xy_ports = [2, 3, 2, 3, 2, 3, 2, 3]
connectivity = Connectivity()
for i in range(8):
    connectivity.add_resonator_line(qubits=qubits[i], constraints=mw_fem_spec(con=1, slot=rr_slots[i], in_port=rr_in_ports[i], out_port=rr_out_ports[i]))
    connectivity.add_qubit_drive_lines(qubits=qubits[i], constraints=mw_fem_spec(con=1, slot=xy_slots[i], out_port=xy_ports[i]))

qubit_pairs = [(i+1, i+2) for i in range(7)]
qubit_pairs.append((8,1))
control_qubits = [c for c, t in qubit_pairs] # get opx_iq_octave_spec from connectivity
target_qubits = [t for c, t in qubit_pairs] # get opx_iq_octave_spec from connectivity
control_qubit_constraints = len(control_qubits) * [opx_iq_octave_spec] # control_qubits
target_qubit_constraints = len(target_qubits) * [opx_iq_octave_spec] # target_qubits

    # connectivity.add_qubit_pair_control_lines(qubit_pairs=qp, constraints=cqc)
    # connectivity.add_qubit_pair_target_lines(qubit_pairs=qp, constraints=tqc)

# Allocate the wiring to the connectivity object based on the available instruments
allocate_wiring(connectivity, instruments, block_used_channels=False)
for i in range(len(qubit_pairs)):
    connectivity.add_qubit_pair_cross_resonance_lines(qubit_pairs=qubit_pairs[i], constraints=mw_fem_spec(con=1, slot=xy_slots[i], out_port=xy_ports[i]))
    allocate_wiring(connectivity, instruments, block_used_channels=False)
    connectivity.add_qubit_pair_zz_drive_lines(qubit_pairs=qubit_pairs[i], constraints=mw_fem_spec(con=1, slot=xy_slots[i], out_port=xy_ports[i]))
    allocate_wiring(connectivity, instruments, block_used_channels=False)
# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path, QuAM)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.figure()
plt.close()

# %%

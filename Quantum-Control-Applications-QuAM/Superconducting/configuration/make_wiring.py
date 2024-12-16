# %%
from pathlib import Path

from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_libs.quam_builder.machine import build_quam_wiring


# Define static parameters
host_ip = "127.0.0.1"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "./quam_state"

# Define the available instrument setup
instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_octave(indices=1)

# Define which quantum elements are present in the system
qubits = [1, 2]
connectivity = Connectivity()
connectivity.add_resonator_line(qubits=qubits)
connectivity.add_qubit_drive_lines(qubits=qubits)

qubit_pairs = [
    (1, 2), (2, 1),
    (2, 3), (3, 2),
]
control_qubits = [c for c, t in qubit_pairs] # get opx_iq_octave_spec from connectivity
target_qubits = [t for c, t in qubit_pairs] # get opx_iq_octave_spec from connectivity
control_qubit_constraints = len(control_qubits) * [opx_iq_octave_spec] # control_qubits
target_qubit_constraints = len(target_qubits) * [opx_iq_octave_spec] # target_qubits
for qp, cqc, tqc in zip(qubit_pairs, control_qubit_constraints, target_qubit_constraints):
    connectivity.add_qubit_pair_control_lines(qubit_pairs=qp, constraints=cqc)
    connectivity.add_qubit_pair_target_lines(qubit_pairs=qp, constraints=tqc)

# Allocate the wiring to the connectivity object based on the available instruments
allocate_wiring(connectivity, instruments)

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)

# %%

# %%
from qualang_tools.wirer.wirer.channel_specs import opx_spec, octave_spec, opx_iq_octave_spec, opx_iq_ext_mixer_spec
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_libs.quam_builder.machine import build_quam_wiring

# Define static parameters
host_ip = "172.16.33.101"  # QOP IP address
port = None  # QOP Port
cluster_name = "Cluster_83"  # Name of the cluster
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "./quam_state"

# Define the available instrument setup
instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_external_mixer(indices=[1, 2])

# Define which qubit indices are present in the system
qubits = [1, 2, 3]
# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()

# Single feed-line for reading the resonators & individual qubit drive lines

# Define any custom/hardcoded channel addresses
# q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)  # pass to the `constraints` argument
# connectivity.add_resonator_line(qubits=qubits, constraints=None)
# connectivity.add_qubit_flux_lines(qubits=qubits)
# connectivity.add_qubit_drive_lines(qubits=qubits)
# # connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(1,2)])  # Tunable coupler
# allocate_wiring(connectivity, instruments)

# Single feed-line for reading the resonators & driving the qubits + flux on specific fem slot
for qubit in qubits:
    connectivity.add_qubit_drive_lines(qubits=qubit, constraints=opx_iq_ext_mixer_spec(out_port_i=3, out_port_q=4))
    allocate_wiring(connectivity, instruments, block_used_channels=qubit==3)

connectivity.add_resonator_line(qubits=qubits, constraints=None)
connectivity.add_qubit_flux_lines(qubits=qubits)
allocate_wiring(connectivity, instruments)
# connectivity.add_qubit_drive_lines(qubits=qubits)
# connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(1,2)])  # Tunable coupler

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)

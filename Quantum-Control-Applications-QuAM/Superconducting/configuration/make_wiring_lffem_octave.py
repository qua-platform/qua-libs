# %%
from qualang_tools.wirer.wirer.channel_specs import lf_fem_spec, lf_fem_iq_octave_spec, octave_spec
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_libs.quam_builder.machine import build_quam_wiring

# Define static parameters
host_ip = "127.0.0.1"  # QOP IP address
port = None  # QOP Port
cluster_name = "Cluster_1"  # Name of the cluster
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "./quam_state"

# Define the available instrument setup
instruments = Instruments()
instruments.add_lf_fem(controller=1, slots=[1, 2])
instruments.add_octave(indices=1)

# Define which quantum elements are present in the system
qubits = [1, 2, 3]
# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()

# Single feed-line for reading the resonators & individual qubit drive lines
# Define any custom/hardcoded channel addresses
q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)
connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
connectivity.add_qubit_flux_lines(qubits=qubits)
connectivity.add_qubit_drive_lines(qubits=qubits)
# connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(1,2), (2,3)])  # Tunable coupler
allocate_wiring(connectivity, instruments)

# Single feed-line for reading the resonators & driving the qubits + flux on specific fem slot
# Define any custom/hardcoded channel addresses
# q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)
# q1_drive_ch = lf_fem_iq_octave_spec(con=1, slot=1, out_port_i=3, out_port_q=4, octave_index=1, rf_out=2)
# q1_flux_con = lf_fem_spec(con=1, out_slot=2, out_port=None)
# connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
# connectivity.add_qubit_flux_lines(qubits=qubits, constraints=q1_flux_con)
# # connectivity.add_qubit_pair_flux_lines(qubit_pairs=[(1,2)])  # Tunable coupler
# for qubit in qubits:
#     connectivity.add_qubit_drive_lines(qubits=qubit, constraints=q1_drive_ch)
#     allocate_wiring(connectivity, instruments, block_used_channels=False)

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path, port)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)

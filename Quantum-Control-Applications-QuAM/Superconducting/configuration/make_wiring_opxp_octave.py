# %%
from qualang_tools.wirer.wirer.channel_specs import opx_spec, octave_spec, opx_iq_octave_spec
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_libs.quam_builder.machine import build_quam_wiring

# Define static parameters
host_ip = "172.16.33.101"  # QOP IP address
cluster_name = "Cluster_81"  # Name of the cluster
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "./quam_state"

# Define the available instrument setup
instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_octave(indices=1)

# Define which qubit indices are present in the system
qubits = [1, 2]
# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()

# Single feed-line for reading the resonators & individual qubit drive lines
# Define any custom/hardcoded channel addresses
q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)
connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
connectivity.add_qubit_drive_lines(qubits=[1], constraints=octave_spec(index=1, rf_out=2))
connectivity.add_qubit_drive_lines(qubits=[2], constraints=octave_spec(index=1, rf_out=3))
connectivity.add_qubit_pair_cross_drive_lines(qubit_pairs=[(1,2),(2,1)])  # Tunable coupler
allocate_wiring(connectivity, instruments)

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)

# %%

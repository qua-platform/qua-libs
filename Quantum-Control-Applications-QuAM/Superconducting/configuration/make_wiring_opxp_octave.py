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
qubits = [1, 2, 3, 4]
qubit_pairs = [
    (1, 2), (2, 1),
    (2, 3), (3, 2),
    (3, 4), (4, 3),
]
# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()

# Single feed-line for reading the resonators & individual qubit drive lines
# Define any custom/hardcoded channel addresses
q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)
qs_xy_ch_map = {
    1: octave_spec(index=1, rf_out=2),
    2: octave_spec(index=1, rf_out=3),
    3: octave_spec(index=1, rf_out=4),
    4: octave_spec(index=1, rf_out=5),
}

connectivity.add_resonator_line(qubits=qubits)
allocate_wiring(connectivity, instruments)

connectivity.add_qubit_drive_lines(qubits=[1, 2, 3, 4])
allocate_wiring(connectivity, instruments, block_used_channels=False)

for qp in qubit_pairs:
    connectivity.add_qubit_pair_cross_resonance_lines(
        qubit_pairs=[qp],
        constraints=qs_xy_ch_map[qp[0]],
    )  # Cross Resonance
    allocate_wiring(connectivity, instruments, block_used_channels=False)
    connectivity.add_qubit_pair_zz_drive_lines(
        qubit_pairs=[qp],
        constraints=qs_xy_ch_map[qp[0]],
    )  # Cross Resonance
    allocate_wiring(connectivity, instruments, block_used_channels=False)

# connectivity.add_qubit_pair_zz_drive_lines(qubit_pairs=[(2,1)], constraints=q2_xy_ch)  # Cross Drive
# allocate_wiring(connectivity, instruments)

# connectivity.add_flux_tunable_transmons(qubits=[1,2])
# allocate_wiring(connectivity, instruments)

# Build the wiring and network into a QuAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path)

# View wiring schematic
# visualize(connectivity.elements, available_channels=instruments.available_channels)

# %%

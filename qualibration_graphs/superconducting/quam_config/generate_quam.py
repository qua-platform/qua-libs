"""
General purpose script to generate the wiring and the QUAM that corresponds to your experiment for the first time.
The workflow is as follows:
    - Copy the content of the wiring example corresponding to your architecture and paste it here.
    - Modify the statis parameters to match your network configuration.
    - Update the instrument setup section with the available hardware.
    - Define which qubit ids are present in the system.
    - Define any custom/hardcoded channel addresses.
    - Allocate the wiring to the connectivity object based on the available instruments.
    - Visualize and validate the resulting connectivity.
    - Build the wiring and QUAM.
    - Populate the generated quam with initial values by modifying and running populate_quam_xxx.py
"""

import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.superconducting import build_quam
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "172.16.33.115"  # QOP IP address
port = None  # QOP Port
cluster_name = "CS_3"  # Name of the cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1, 2, 3, 4])
instruments.add_lf_fem(controller=1, slots=[5, 6, 7, 8])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [0, 1, 2, 3, 4, 5, 6, 7, 8]
couplers = []

# Grid size
rows = 3
cols = 3

for i in range(rows):
    for j in range(cols):
        index = i * cols + j

        # Horizontal coupler (to right neighbor)
        if j < cols - 1:
            right = i * cols + (j + 1)
            couplers.append((index, right))

        # Vertical coupler (to bottom neighbor)
        if i < rows - 1:
            down = (i + 1) * cols + j
            couplers.append((index, down))


########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# multiplexed readout for qubits {0, 3, 6}, {1, 4, 7} and {2, 5, 8} on three feed-lines
q036_res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
q147_res_ch = mw_fem_spec(con=1, slot=2, in_port=1, out_port=1)
q258_res_ch = mw_fem_spec(con=1, slot=3, in_port=1, out_port=1)
# individual xy drive for qubits {0, 3, 6} on MW-FEM 1
q036_drive_ch = mw_fem_spec(con=1, slot=1, in_port=None, out_port=None)
# individual xy drive for qubits {1, 4, 7} on MW-FEM 2
q147_drive_ch = mw_fem_spec(con=1, slot=2, in_port=None, out_port=None)
# individual xy drive for qubits {2, 5, 8} on MW-FEM 3
q258_drive_ch = mw_fem_spec(con=1, slot=3, in_port=None, out_port=None)


########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# The readout lines
connectivity.add_resonator_line(qubits=qubits[0::3], constraints=q036_res_ch)
connectivity.add_resonator_line(qubits=qubits[1::3], constraints=q147_res_ch)
connectivity.add_resonator_line(qubits=qubits[2::3], constraints=q258_res_ch)

# The xy drive lines
connectivity.add_qubit_drive_lines(qubits=qubits[0::3], constraints=q036_drive_ch)
connectivity.add_qubit_drive_lines(qubits=qubits[1::3], constraints=q147_drive_ch)
connectivity.add_qubit_drive_lines(qubits=qubits[2::3], constraints=q258_drive_ch)

# The flux lines for the individual qubits
connectivity.add_qubit_flux_lines(qubits=qubits)
# The flux lines for the tunable couplers
connectivity.add_qubit_pair_flux_lines(qubit_pairs=couplers)
# Allocate the wiring
allocate_wiring(connectivity, instruments)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=False)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
# user_input = input("Do you want to save the updated QUAM? (y/n)")
# if user_input.lower() == "y":
machine = Quam()
# Build the wiring (wiring.json) and initiate the QUAM
build_quam_wiring(connectivity, host_ip, cluster_name, machine)

# Reload QUAM, build the QUAM object and save the state as state.json
machine = Quam.load()
# print("QUAM loaded from:", machine._path)
build_quam(machine)

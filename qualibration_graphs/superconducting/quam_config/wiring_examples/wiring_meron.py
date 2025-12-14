import matplotlib.pyplot as plt
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize
from qualang_tools.wirer.wirer.channel_specs import *
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
instruments.add_mw_fem(controller=1, slots=[1, 2])
instruments.add_lf_fem(controller=1, slots=[5, 7, 4])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = ["B1", "B2", "B3", "A4"]
qubit_pairs = [("B1", "B2"), ("B2", "B3"), ("B3", "A4"), ("A4", "B1")]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# multiplexed readout for qubits 1 to 4 and 5 to 8 on two feed-lines
res_ch = mw_fem_spec(con=1, slot=2, in_port=1, out_port=1)
res_chA = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
# qB2_res_ch = mw_fem_spec(con=1, slot=2, in_port=1, out_port=1)
# individual xy drive for qubits 1 to 4 on MW-FEM 1
qB1_drive_ch = mw_fem_spec(con=1, slot=2, in_port=None, out_port=2)
qB2_drive_ch = mw_fem_spec(con=1, slot=2, in_port=None, out_port=3)
qB3_drive_ch = mw_fem_spec(con=1, slot=2, in_port=None, out_port=4)
qA4_drive_ch = mw_fem_spec(con=1, slot=1, in_port=None, out_port=5)

qB1_flux_ch = lf_fem_spec(con=1, out_slot=5, out_port=1)
qB2_flux_ch = lf_fem_spec(con=1, out_slot=5, out_port=2)
qB3_flux_ch = lf_fem_spec(con=1, out_slot=5, out_port=3)
qA4_flux_ch = lf_fem_spec(con=1, out_slot=4, out_port=4)

coupler_chb1b2 = lf_fem_spec(con=1, out_slot=7, out_port=7)
coupler_chb2b3 = lf_fem_spec(con=1, out_slot=7, out_port=8)
coupler_chb3a4 = lf_fem_spec(con=1, out_slot=7, out_port=6)
coupler_cha4b1 = lf_fem_spec(con=1, out_slot=7, out_port=5)

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# The readout lines
connectivity.add_resonator_line(qubits=qubits[:-1], constraints=res_ch)
connectivity.add_resonator_line(qubits=qubits[-1], constraints=res_chA)
allocate_wiring(connectivity, instruments, block_used_channels=True)

# The xy drive lines
connectivity.add_qubit_drive_lines(qubits=qubits[0], constraints=qB1_drive_ch)
connectivity.add_qubit_drive_lines(qubits=qubits[1], constraints=qB2_drive_ch)
connectivity.add_qubit_drive_lines(qubits=qubits[2], constraints=qB3_drive_ch)
connectivity.add_qubit_drive_lines(qubits=qubits[3], constraints=qA4_drive_ch)

allocate_wiring(connectivity, instruments, block_used_channels=True)

connectivity.add_qubit_flux_lines(qubits=qubits[0], constraints=qB1_flux_ch)
connectivity.add_qubit_flux_lines(qubits=qubits[1], constraints=qB2_flux_ch)
connectivity.add_qubit_flux_lines(qubits=qubits[2], constraints=qB3_flux_ch)
connectivity.add_qubit_flux_lines(qubits=qubits[3], constraints=qA4_flux_ch)

allocate_wiring(connectivity, instruments, block_used_channels=True)

# The flux lines for the tunable couplers
connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs[0], constraints=coupler_chb1b2)
connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs[1], constraints=coupler_chb2b3)
connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs[2], constraints=coupler_chb3a4)
connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs[3], constraints=coupler_cha4b1)
# Allocate the wiring
allocate_wiring(connectivity, instruments)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels, use_matplotlib=True)
plt.show(block=False)

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

# %%

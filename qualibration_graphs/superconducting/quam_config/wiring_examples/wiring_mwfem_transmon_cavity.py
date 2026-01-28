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
calibration_db_path = None  # "/path/to/some/config/folder"

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [1]  # Single transmon with cavity

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# Readout resonator on MW-FEM slot 1, port 1
rr_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
# Transmon XY drive on MW-FEM slot 1, port 2
xy_ch = mw_fem_spec(con=1, slot=1, out_port=2)
# Cavity drive on MW-FEM slot 1, port 3
cavity_ch = mw_fem_spec(con=1, slot=1, out_port=3)
# TWPA pump on MW-FEM slot 1, port 4 (dedicated output for parametric amplification)
twpa_ch = mw_fem_spec(con=1, slot=1, out_port=4)

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# The readout line
connectivity.add_resonator_line(qubits=qubits, constraints=rr_ch)
# The transmon XY drive line
connectivity.add_qubit_drive_lines(qubits=qubits, constraints=xy_ch)
# The cavity line (single qubit only - each cavity is associated with exactly one transmon)
connectivity.add_cavity_lines(qubit=1, constraints=cavity_ch)
# TWPA pump line (standalone element for readout signal amplification)
connectivity.add_twpa_lines(twpas=[1], constraints=twpa_ch)
# Allocate the wiring
allocate_wiring(connectivity, instruments)

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
    build_quam(machine, calibration_db_path)

# %%

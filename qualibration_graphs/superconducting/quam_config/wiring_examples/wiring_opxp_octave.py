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
calibration_db_path = None  # "/path/to/some/config/folder"

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_opx_plus(controllers=[1, 2])
instruments.add_octave(indices=1)

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
qubits = [1, 2, 3, 4]
qubit_pairs = [(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# multiplexed readout for qubits 1 to 5
q1_res_ch = octave_spec(index=1, rf_out=1, rf_in=1)

########################################################################################################################
# %%                 Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# The readout line
connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
# The individual xy drive lines
connectivity.add_qubit_drive_lines(qubits=qubits)
# The flux lines for the individual qubits
connectivity.add_qubit_flux_lines(qubits=qubits)
# The flux lines for the tunable couplers
connectivity.add_qubit_pair_flux_lines(qubit_pairs=qubit_pairs)
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

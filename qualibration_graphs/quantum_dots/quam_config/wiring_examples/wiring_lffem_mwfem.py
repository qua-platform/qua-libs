import matplotlib.pyplot as plt
from qualang_tools.wirer.wirer.channel_specs import *
from qualang_tools.wirer import Instruments, Connectivity, allocate_wiring, visualize
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.quantum_dots import build_quam
from quam_builder.architecture.quantum_dots.operations.macro_catalog import VoltageBalancedMacroCatalog
from quam_config import Quam

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "172.16.33.115"  # QOP IP address
port = None  # QOP Port
cluster_name = "CS_4"  # Name of the cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1])
instruments.add_lf_fem(controller=1, slots=[5, 6])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
quantum_dots=[1, 2, 3, 4]
sensor_dots=[1, 2]
quantum_dot_pairs=[(1, 2), (2, 3), (3, 4)]
qubits = [1, 2, 3, 4]
qubit_pairs = [(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# multiplexed readout for qubits 1 to 4 and 5 to 8 on two feed-lines
q1to4_res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
q5to8_res_ch = mw_fem_spec(con=1, slot=2, in_port=1, out_port=1)
# individual xy drive for qubits 1 to 4 on MW-FEM 1
q1to4_drive_ch = mw_fem_spec(con=1, slot=1, in_port=None, out_port=None)
# multiplexed xy drive for qubits 5 to 8 on MW-FEM 2 port 4
q5to8_drive_ch = mw_fem_spec(con=1, slot=2, in_port=None, out_port=4)

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# Add the plunger gates and drive lines for each dot
connectivity.add_quantum_dots(quantum_dots, add_drive_lines=True, use_mw_fem=True, shared_drive_line=False)
# Add the sensor gates and rf-reflectometry readout components for each sensor dot
connectivity.add_sensor_dots(sensor_dots, shared_resonator_line=False)
# Add the barrier gates for each quantum dot pair
connectivity.add_quantum_dot_pairs(quantum_dot_pairs)
# Allocate the wiring
allocate_wiring(connectivity, instruments)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
plt.show(block=False)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
user_input = input("Do you want to save the updated QUAM? (y/n)")
if user_input.lower() == "y":
    machine = Quam()
    # Build the wiring (wiring_old.json) and initiate the QUAM
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)
    build_quam(machine, catalogs = [VoltageBalancedMacroCatalog()])
    machine.save()
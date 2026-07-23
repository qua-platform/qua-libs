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
host_ip = "127.0.0.1"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1])
instruments.add_lf_fem(controller=1, slots=[5, 6])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
quantum_dots = [1, 2, 3, 4]
sensor_dots = [1, 2]
quantum_dot_pairs = [(1, 2), (2, 3), (3, 4)]
qubits = [1, 2, 3, 4]
qubit_pairs = [(qubits[i], qubits[i + 1]) for i in range(len(qubits) - 1)]

# # Example: map qubit pairs to specific sensor dots (supports multiple sensors per pair).
# # Pair keys: q1_q2 or q1-2. Sensor ids: virtual_sensor_<n>, sensor_<n>, or s<n> (e.g., virtual_sensor_1, sensor_1, s1).
qubit_pair_sensor_map = {
    "q1_q2": ["sensor_1"],
    "q2_q3": ["sensor_1", "sensor_2"],
    "q3_q4": ["sensor_2"],
}

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# Add the plunger gates and drive lines for each dot
connectivity.add_quantum_dots(quantum_dots, add_drive_lines=True, use_mw_fem=True, shared_drive_line=True)
# Add the sensor gates and rf-reflectometry readout components for each sensor dot with the constraint of being on the 2nd LF-FEM
connectivity.add_sensor_dots(sensor_dots, shared_resonator_line=False, constraints=lf_fem_spec(out_slot=6, in_slot=6))
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
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)
    build_quam(machine, qubit_pair_sensor_map=qubit_pair_sensor_map, catalogs=[VoltageBalancedMacroCatalog()])
    machine.save()

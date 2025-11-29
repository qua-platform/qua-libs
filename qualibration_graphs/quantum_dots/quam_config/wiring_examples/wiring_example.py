import matplotlib.pyplot as plt
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize

from qualang_tools.wirer.connectivity.wiring_spec import (
    WiringFrequency,
    WiringIOType,
    WiringLineType,
)
from qualang_tools.wirer.wirer.channel_specs import *
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.quantum_dots import build_quam

# from quam_config import Quam
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
instruments.add_lf_fem(controller=1, slots=[3, 4, 5])

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
global_gates = [1, 2]
sensor_dots = [1, 2]
qubits = [1, 2, 3, 4]
qubit_pairs = [(1, 2), (2, 3), (3, 4)]

########################################################################################################################
# %%                                 Define any custom/hardcoded channel addresses
########################################################################################################################
# multiplexed readout for sensor 1 to 2 and 3 to 4 on two feed-lines
# s1to2_res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
# s3to4_res_ch = mw_fem_spec(con=1, slot=2, in_port=1, out_port=1)

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
# The readout lines
connectivity.add_voltage_gate_lines(voltage_gates=global_gates, name="rb")

# Option 1
connectivity.add_sensor_dots(sensor_dots=sensor_dots, shared_resonator_line=False)

# Option 2
# connectivity.add_sensor_dot_resonator_line(sensor_dots, wiring_frequency=WiringFrequency.DC)
# connectivity.add_sensor_dot_voltage_gate_lines(sensor_dots)

# Option 1:
connectivity.add_qubits(qubits=qubits)
# Option 2:
# connectivity.add_qubit_voltage_gate_lines(qubits)
# connectivity.add_quantum_dot_qubit_drive_lines(qubits, wiring_frequency=WiringFrequency.DC)

connectivity.add_qubit_pairs(qubit_pairs=qubit_pairs)
allocate_wiring(connectivity, instruments)

# Optional: visualize wiring (requires a GUI backend). Comment out in headless environments.
import matplotlib
matplotlib.use("TkAgg")
visualize(
    connectivity.elements,
    available_channels=instruments.available_channels,
    use_matplotlib=True,
)
plt.show()

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
machine = BaseQuamQD()

machine = build_quam_wiring(
    connectivity,
    host_ip,
    cluster_name,
    machine,
)

machine.generate_config()

# Example: map qubit pairs to specific sensor dots (supports multiple sensors per pair).
# Pair keys: q1_q2 or q1-2. Sensor ids: virtual_sensor_<n>, sensor_<n>, or s<n> (e.g., virtual_sensor_1, sensor_1, s1).
qubit_pair_sensor_map = {
    "q1_q2": ["s1"],
    "q2_q3": ["s1", "sensor_2"],
    "q3_q4": ["s2"],
}

build_quam(machine, qubit_pair_sensor_map=qubit_pair_sensor_map)
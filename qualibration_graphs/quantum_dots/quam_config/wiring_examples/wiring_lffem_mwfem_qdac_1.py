import matplotlib.pyplot as plt
from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize

from qualang_tools.wirer.wirer.channel_specs import *

from quam_builder.architecture.quantum_dots.operations.macro_catalog import VoltageBalancedMacroCatalog
from quam_config import Quam
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.quantum_dots import build_quam

QUAM_STATE_PATH = "/Users/kalidu_laptop/merge_libs/quam_state"

def qdac_config(ip: str):
    return {
        "driver_module": "qcodes_contrib_drivers.drivers.QDevil.QDAC2",
        "driver_class": "QDac2",
        "connection": {
            "visalib": "@py",
            "address": f"TCPIP::{ip}::5025::SOCKET",
        },
        "channel_method": "channel",
        "accessor": "limited_dc_constant_V",
        "is_qdac": True,
    }

########################################################################################################################
# %%                                              Define static parameters
########################################################################################################################
host_ip = "172.16.33.115"  # QOP IP address
port = None  # QOP Port
cluster_name = "CS_4"  # Name of the cluster
# QDAC IP addresses
qdac_ips = ["172.16.33.101", "172.16.33.101"]
dac_config = {}
for i, qdac in enumerate(qdac_ips):
    dac_config[f"qdac{i + 1}"] = qdac_config(qdac_ips[i])

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_lf_fem(controller=1, slots=[5, 6])
instruments.add_mw_fem(controller=1, slots=[1])
instruments.add_qdac2(indices=[1,2])

########################################################################################################################
# %%                           Define which quantum elements are present in the system
########################################################################################################################
plunger_dots = [1, 2, 3, 4]  # P1, P2
sensor_dots = [1, 2]
# global_gates = [1]

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

# Add plunger gates
# Given the constraints that we would like to put on the outputs (i.e. which dot is from which channel), we add them individually with their own constraint, 
# Rather than using connectivity.add_quantum_dots(plunger_dots)
qdac_lf_spec = qdac2_spec(1) & lf_fem_spec(1)
connectivity.add_quantum_dot_voltage_gate_lines(1, True, constraints=lf_fem_spec(1) & qdac2_spec(index = 1, out_port = 1, trigger_in_port=1))
connectivity.add_quantum_dot_voltage_gate_lines(2, True, lf_fem_spec(1) & qdac2_spec(index = 2, out_port = 2, trigger_in_port=2))
connectivity.add_quantum_dot_voltage_gate_lines(3, True, qdac_lf_spec)
connectivity.add_quantum_dot_voltage_gate_lines(4, constraints=qdac_lf_spec)
# Add global gates
connectivity.add_voltage_gate_lines("source", name="", constraints=qdac2_spec(out_port=10))
connectivity.add_voltage_gate_lines("drain", name="", constraints=qdac2_spec(out_port=11))
# Add sensor dots
connectivity.add_sensor_dot_voltage_gate_lines(sensor_dots, constraints=qdac_lf_spec)
# Add resonators
connectivity.add_sensor_dot_resonator_line(sensor_dots, shared_line=True)
# Add drive lines
connectivity.add_quantum_dot_drive_lines(plunger_dots, shared_line=True, use_mw_fem=True)
# Allocate the wiring
allocate_wiring(connectivity, instruments)

# Optional: visualize wiring (requires a GUI backend). Comment out in headless environments.
visualize(connectivity.elements, instruments.available_channels)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################

user_input = input("Do you want to save the updated QUAM? (y/n)")
if user_input.lower() == "y":
    machine = Quam()

    build_quam_wiring(
        connectivity,
        host_ip,
        cluster_name,
        machine,
        dac_config=dac_config,
        path=QUAM_STATE_PATH,
    )
    build_quam(
        machine,
        qubit_pair_sensor_map=qubit_pair_sensor_map,
        save=False,
        connect_qdac=False,
        catalogs = [VoltageBalancedMacroCatalog()],
    )

    machine.save(QUAM_STATE_PATH)

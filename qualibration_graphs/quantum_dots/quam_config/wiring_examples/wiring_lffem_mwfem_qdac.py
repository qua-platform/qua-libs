from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring, visualize

from qualang_tools.wirer.wirer.channel_specs import *

from quam_builder.architecture.quantum_dots.operations.macro_catalog import VoltageBalancedMacroCatalog
from quam_config import Quam
from quam_builder.builder.qop_connectivity import build_quam_wiring
from quam_builder.builder.quantum_dots import build_quam


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
# QOP network setting
host_ip = "127.0.0.1"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster

# QDAC IP addresses
qdac_ips = ["127.0.0.2", "127.0.0.3"]
dac_config = {}
for i, qdac in enumerate(qdac_ips):
    dac_config[f"qdac{i + 1}"] = qdac_config(qdac_ips[i])

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_mw_fem(controller=1, slots=[1])
instruments.add_lf_fem(controller=1, slots=[5, 6])
instruments.add_qdac2(indices=[1, 2])

########################################################################################################################
# %%                           Define which quantum elements are present in the system
########################################################################################################################
plunger_dots = [1, 2, 3, 4]  # P1, P2
sensor_dots = [1, 2]

# Quantum Dot Pairs defines the Barrier Gates
quantum_dot_pairs = [(plunger_dots[i], plunger_dots[i + 1]) for i in range(len(plunger_dots) - 1)]

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

# The first arg for the spec is the INDEX, i.e. which controller you are referring to. In order to auto-allocate channels from the specified controller, we can
# make a convenience variable here to combine the OPX CON1 & QDAC1
qdac_lf_spec = qdac2_spec(1) & lf_fem_spec(1)

connectivity.add_quantum_dot_voltage_gate_lines(1, True, lf_fem_spec(1) & qdac2_spec(out_port=1, trigger_in_port=1))
connectivity.add_quantum_dot_voltage_gate_lines(2, True, lf_fem_spec(1) & qdac2_spec(out_port=2, trigger_in_port=2))
connectivity.add_quantum_dot_voltage_gate_lines(3, False, qdac_lf_spec)
connectivity.add_quantum_dot_voltage_gate_lines(4, constraints=qdac_lf_spec)
# Add global gates
connectivity.add_voltage_gate_lines("source", name="", constraints=qdac2_spec(index=2, out_port=10))
connectivity.add_voltage_gate_lines("drain", name="", constraints=qdac2_spec(index=2, out_port=11))
# Add sensor dots
connectivity.add_sensor_dot_voltage_gate_lines(sensor_dots, constraints=qdac_lf_spec)
# Add resonators
connectivity.add_sensor_dot_resonator_line(sensor_dots, shared_line=False)
# Add drive lines
connectivity.add_quantum_dot_drive_lines(plunger_dots, shared_line=True, use_mw_fem=True)
# Add the barrier gates for each quantum dot pair
connectivity.add_quantum_dot_pairs(quantum_dot_pairs, constraints=lf_fem_spec(out_slot=6) & qdac2_spec(index=1))
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
    )
    build_quam(
        machine,
        qubit_pair_sensor_map=qubit_pair_sensor_map,
        catalogs=[VoltageBalancedMacroCatalog()],
        connect_qdac=False,
    )
    machine.save("/Users/kalidu_laptop/merge_libs/quam_state")
    machine.save()

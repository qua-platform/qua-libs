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
instruments.add_opx_plus(controllers=[1])
instruments.add_octave(indices=1)

########################################################################################################################
# %%                                 Define which qubit ids are present in the system
########################################################################################################################
quantum_dots = [1, 2, 3]
sensor_dots = [1]
quantum_dot_pairs = [(1, 2), (2, 3)]

# Example: map qubit pairs to specific sensor dots (supports multiple sensors per pair).
qubit_pair_sensor_map = {
    "q1_q2": ["sensor_1"],
    "q2_q3": ["sensor_1"],
}

########################################################################################################################
# %%                                 Define OPX+ / Octave channel constraints
########################################################################################################################
# Sensor reflectometry: OPX+ input/output resonator line (no Octave on readout).
# Pins the ADC port; the wirer auto-allocates a matching OPX+ output port.
sensor_1_readout = opx_spec(con=1, in_port=1)

# Shared spin-qubit ESR drive: OPX+ baseband IQ → Octave upconversion → RF output.
drive_ch = opx_iq_octave_spec(con=1, rf_out=1)

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()
connectivity.add_quantum_dots(
    quantum_dots,
    add_drive_lines=True,
    use_mw_fem=True,
    shared_drive_line=True,
    constraints=drive_ch,
)
connectivity.add_sensor_dot_voltage_gate_lines(sensor_dots)
connectivity.add_sensor_dot_resonator_line(
    1,
    shared_line=False,
    use_mw_fem=False,
    constraints=sensor_1_readout,
)
connectivity.add_quantum_dot_pairs(quantum_dot_pairs)

allocate_wiring(connectivity, instruments)

visualize(connectivity.elements, available_channels=instruments.available_channels)

########################################################################################################################
# %%                                   Build the wiring and QUAM
########################################################################################################################
user_input = input("Do you want to save the updated QUAM? (y/n)")
if user_input.lower() == "y":
    machine = Quam()
    build_quam_wiring(connectivity, host_ip, cluster_name, machine)
    build_quam(
        machine,
        qubit_pair_sensor_map=qubit_pair_sensor_map,
        catalogs=[VoltageBalancedMacroCatalog()],
    )
    machine.save()

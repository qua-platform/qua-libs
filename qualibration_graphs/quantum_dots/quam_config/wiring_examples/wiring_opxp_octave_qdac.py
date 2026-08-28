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
host_ip = "127.0.0.1"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster

# QDAC IP addresses
qdac_ips = ["127.0.0.2"]
dac_config = {}
for i, qdac in enumerate(qdac_ips):
    dac_config[f"qdac{i + 1}"] = qdac_config(qdac_ips[i])

########################################################################################################################
# %%                                      Define the available instrument setup
########################################################################################################################
instruments = Instruments()
instruments.add_opx_plus(controllers=[1])
instruments.add_octave(indices=1)
instruments.add_qdac2(indices=[1])

########################################################################################################################
# %%                           Define which quantum elements are present in the system
########################################################################################################################
plunger_dots = [1, 2, 3]
sensor_dots = [1]

quantum_dot_pairs = [(plunger_dots[i], plunger_dots[i + 1]) for i in range(len(plunger_dots) - 1)]

qubit_pair_sensor_map = {
    "q1_q2": ["sensor_1"],
    "q2_q3": ["sensor_1"],
}

########################################################################################################################
# %%                                 Define OPX+ / Octave / QDAC channel constraints
########################################################################################################################
# Convenience spec: QDAC1 coarse bias + OPX+ fine/fast outputs on controller 1.
qdac_opx_spec = qdac2_spec(1) & opx_spec(con=1)

# Sensor reflectometry: OPX+ input/output resonator line (no Octave on readout).
sensor_1_readout = opx_spec(con=1, in_port=1)

# Shared spin-qubit ESR drive: OPX+ baseband IQ → Octave RF output.
drive_ch = opx_iq_octave_spec(con=1, rf_out=1)

########################################################################################################################
# %%                Allocate the wiring to the connectivity object based on the available instruments
########################################################################################################################
connectivity = Connectivity()

# Plunger gates — same per-dot constraint pattern as wiring_lffem_mwfem_qdac.py,
# with lf_fem_spec replaced by opx_spec(con=1).
connectivity.add_quantum_dot_voltage_gate_lines(
    1, True, opx_spec(con=1) & qdac2_spec(index=1, out_port=1, trigger_in_port=1)
)
connectivity.add_quantum_dot_voltage_gate_lines(
    2, True, opx_spec(con=1) & qdac2_spec(index=1, out_port=2, trigger_in_port=2)
)
connectivity.add_quantum_dot_voltage_gate_lines(3, False, qdac_opx_spec)

# Global gates on QDAC2
connectivity.add_voltage_gate_lines("source", name="", constraints=qdac2_spec(index=1, out_port=10))
connectivity.add_voltage_gate_lines("drain", name="", constraints=qdac2_spec(index=1, out_port=11))

# Sensor gate lines (QDAC coarse + OPX+ fine)
connectivity.add_sensor_dot_voltage_gate_lines(sensor_dots, constraints=qdac_opx_spec)

# Sensor reflectometry (OPX+ input/output; pins in_port, output auto-allocated)
connectivity.add_sensor_dot_resonator_line(
    1,
    shared_line=False,
    use_mw_fem=False,
    constraints=sensor_1_readout,
)

# Shared ESR drive through Octave
connectivity.add_quantum_dot_drive_lines(
    plunger_dots,
    shared_line=True,
    use_mw_fem=True,
    constraints=drive_ch,
)

# Barrier gates (OPX+ fast outputs + QDAC1 bias), analogous to LF-FEM slot 6 in the QDAC example
connectivity.add_quantum_dot_pairs(
    quantum_dot_pairs,
    constraints=opx_spec(con=1) & qdac2_spec(index=1),
)

allocate_wiring(connectivity, instruments)

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
    machine.save()

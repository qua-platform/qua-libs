"""
Generate the QuAM for a parametric-cavity device (MW-FEM only).

Device topology
---------------
- Fixed-frequency transmon at 4.3 GHz
- Readout resonator at 4.5 GHz
- Storage cavity at 6.0 GHz (direct XY drive)
- Parametric flux-line drive carrying coupling tones at mode-difference
  frequencies (0.2, 1.5, 1.7 GHz)

Hardware: OPX1000, single MW-FEM (no LF-FEM).

MW-FEM port allocation (slot 1)
-------------------------------
O1 + I1  (Band 1)  – readout resonator + transmon XY  (upconv = 4.4 GHz)
O2       (Band 2)  – cavity direct XY drive            (upconv = 6.0 GHz)
O4       (Band 1)  – parametric coupling tones          (upconv = 1.6 GHz)

Run this script once to create the initial ``state.json``.
Then run ``populate_quam_parametric_cavity.py`` to set frequencies, powers,
and pulses.
"""

from quam.components.ports.analog_outputs import MWFEMAnalogOutputPort
from quam.components.ports.analog_inputs import MWFEMAnalogInputPort
from quam.components.ports.ports_containers import FEMPortsContainer
from quam.components.pulses import SquarePulse, SquareReadoutPulse

from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorMW,
)
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW

from quam_config.my_quam import (
    Quam,
    ParametricCavityTransmon,
    StorageCavity,
    ParametricDriveMW,
)

########################################################################################################################
# %%                                        Static parameters
########################################################################################################################
# QOP / cluster (from CS_4 dashboard: controller 172.16.33.114:80)
host_ip = "172.16.33.114"
cluster_name = "CS_4"
# QUA ``controllers`` key and port ``controller_id`` must match the name in QOP
# resources (dashboard cluster label CS_4 is *not* the controller id). Typical
# OPX1000 install uses ``con1`` for the first controller.
controller_id = "con1"

CON = controller_id
MW_SLOT = 1

########################################################################################################################
# %%                                        Build ports container
########################################################################################################################
ports = FEMPortsContainer(
    mw_outputs={
        CON: {
            MW_SLOT: {
                1: MWFEMAnalogOutputPort(
                    controller_id=CON, fem_id=MW_SLOT, port_id=1,
                    band=1, upconverter_frequency=4.4e9, full_scale_power_dbm=-11,
                ),
                2: MWFEMAnalogOutputPort(
                    controller_id=CON, fem_id=MW_SLOT, port_id=2,
                    band=2, upconverter_frequency=6.0e9, full_scale_power_dbm=-11,
                ),
                4: MWFEMAnalogOutputPort(
                    controller_id=CON, fem_id=MW_SLOT, port_id=4,
                    band=1, upconverter_frequency=1.6e9, full_scale_power_dbm=-11,
                ),
            },
        },
    },
    mw_inputs={
        CON: {
            MW_SLOT: {
                1: MWFEMAnalogInputPort(
                    controller_id=CON, fem_id=MW_SLOT, port_id=1,
                    band=1, downconverter_frequency=4.4e9,
                ),
            },
        },
    },
)

# Reference strings so channels can share the same port object (O1 is shared
# between readout and transmon XY).
ref_o1 = f"#/ports/mw_outputs/{CON}/{MW_SLOT}/1"
ref_i1 = f"#/ports/mw_inputs/{CON}/{MW_SLOT}/1"
ref_o2 = f"#/ports/mw_outputs/{CON}/{MW_SLOT}/2"
ref_o4 = f"#/ports/mw_outputs/{CON}/{MW_SLOT}/4"

########################################################################################################################
# %%                                  Build channels and qubit
########################################################################################################################
resonator = ReadoutResonatorMW(
    id="rr1",
    opx_output=ref_o1,
    opx_input=ref_i1,
    RF_frequency=4.5e9,
    f_01=4.5e9,
    operations={"readout": SquareReadoutPulse(length=1000, amplitude=0.01)},
)

xy = XYDriveMW(
    id="xy1",
    opx_output=ref_o1,
    RF_frequency=4.3e9,
    operations={"saturation": SquarePulse(length=20_000, amplitude=0.01)},
)

cavity_xy = XYDriveMW(
    id="cavity_xy1",
    opx_output=ref_o2,
    RF_frequency=6.0e9,
    operations={"saturation": SquarePulse(length=20_000, amplitude=0.01)},
)

parametric_drive = ParametricDriveMW(
    id="parametric1",
    opx_output=ref_o4,
    RF_frequency=1.7e9,
    operations={"square": SquarePulse(length=200, amplitude=0.1)},
)

cavity = StorageCavity(id="c1", xy=cavity_xy, f_01=6.0e9)

qubit = ParametricCavityTransmon(
    id="q1",
    f_01=4.3e9,
    xy=xy,
    resonator=resonator,
    cavity=cavity,
    parametric_drive=parametric_drive,
)

########################################################################################################################
# %%                                 Assemble and save the QuAM
########################################################################################################################
machine = Quam(
    qubits={"q1": qubit},
    active_qubit_names=["q1"],
    network={"host": host_ip, "cluster_name": cluster_name, "port": 80},
    ports=ports,
)

machine.save()
print("QuAM saved.  Run populate_quam_parametric_cavity.py next.")

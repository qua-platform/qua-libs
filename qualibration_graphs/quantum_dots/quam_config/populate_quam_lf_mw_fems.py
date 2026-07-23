import numpy as np
from typing import List
from pathlib import Path

from quam_config import QubitQuam as Quam
from quam_builder.architecture.quantum_dots.operations.names import (
    DrivePulseName,
    SingleQubitMacroName,
    VoltagePointName,
)

QUAM_STATE_PATH = None

def _apply_fem_output_port_delays(machine: Quam) -> None:
    """Set per-FEM analog output delays (LF path skew vs MW)."""
    for controller_ports in machine.ports.analog_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = LF_FEM_DELAY_NS

    for controller_ports in machine.ports.mw_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = MW_FEM_DELAY_NS


# Align LF vs MW output timing in the QM pulse config (matches ``quam_factory``).
LF_FEM_DELAY_NS: int = 161
MW_FEM_DELAY_NS: int = 0

machine = Quam.load(QUAM_STATE_PATH)

#######################################
###### Qubits Physical Properties #####
#######################################

# XY / MW-FEM: QuAM uses IF = larmor_frequency - MW_upconverter (see XYDriveMW).
# QM enforces |IF| <= 500 MHz. The old name ``LO`` here was really the *Larmor*
# centre (~9.7 GHz), not the FEM LO; leaving upconverter at ~5 GHz made IF ~4.7 GHz.
larmor_center_hz = 9.5e9
mw_upconverter_hz = larmor_center_hz

qubit_frequencies = [8e9, 8.2e9, 9e9, 9.6e9]

#################################
###### Qubits Points Update #####
#################################

for i, q in enumerate(machine.qubits.values()):
    q.xy.opx_output.band = 3
    # Same params for each qubit for now. Subject to change.
    # q.macros[VoltagePointName.INITIALIZE].update(ramp_duration=2000, hold_duration=200)
    # q.macros[VoltagePointName.MEASURE].update(buffer_duration=240)
    # q.macros[VoltagePointName.EMPTY].update(hold_duration=80)

    # MW FEM LO on this XY line (shared port → same value each iteration is fine).
    q.xy.opx_output.upconverter_frequency = mw_upconverter_hz

    # Absolute drive / Larmor frequency (RF), not the OPX IF.
    q.larmor_frequency = qubit_frequencies[i]

    # Update all the existing pulse names based on enum DrivePulseName
    for name in DrivePulseName: 
        # Ignore any pulses that are not mapped to the qubits (e.g. CROT, which is only mapped to the qubit_pair.)
        x90 = q.xy.operations.get(f"{name}_x90", None)
        x180 = q.xy.operations.get(f"{name}_x180", None)
        if x90 is not None: 
            x90.amplitude = 0.15
        if x180 is not None: 
            x180.amplitude = 0.3

    # Default values
    q.T1 = 1e-6
    q.T2ramsey = 0.5e-6
    q.T2echo = 2e-6

#########################
###### State Points #####
#########################

# ### Example generator method to add some default points. OPTIONAL

# for i, qdp in enumerate(machine.quantum_dot_pairs.values()):
#     qdp.add_point(
#         point_name=VoltagePointName.INITIALIZE,
#         voltages={d.id: (i + 1) * 0.015 for d in qdp.quantum_dots},
#         duration=1000,
#     )
#     qdp.add_point(
#         point_name=VoltagePointName.EMPTY,
#         voltages={d.id: (i + 1) * 0.02 for d in qdp.quantum_dots},
#         duration=1500,
#     )
#     qdp.add_point(
#         point_name=VoltagePointName.MEASURE,
#         voltages={d.id: (i + 1) * 0.025 for d in qdp.quantum_dots},
#         duration=1000,
#     )
#     qdp.add_point(
#         point_name=VoltagePointName.EXCHANGE,
#         voltages={d.id: (i + 1) * -0.025 for d in qdp.quantum_dots},
#         duration=1000,
#     )

##############################
###### Sensor Properties #####
##############################

resonator_frequencies = [250e6, 300e6]
for i, s in enumerate(machine.sensor_dots.values()):
    s.readout_resonator.intermediate_frequency = resonator_frequencies[i]
    s.readout_resonator.operations["readout"].amplitude = 0.02
    s.readout_resonator.operations["readout"].length = 5_000  # 5us

################################
###### Compensation Matrix #####
################################

gate_set_id = next(iter(machine.virtual_gate_sets))

qds = machine.quantum_dots
# Orthogonalise the barriers. Detuning will be another layer.
machine.update_cross_compensation_submatrix(
    virtual_names=["virtual_barrier_1", "virtual_barrier_2", "virtual_barrier_3"],
    channels=[
        qds["virtual_dot_1"].physical_channel,
        qds["virtual_dot_2"].physical_channel,
        qds["virtual_dot_3"].physical_channel,
        qds["virtual_dot_4"].physical_channel,
    ],
    matrix=[
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
        [0.2, 0.1, 0.3],
        [0.1, 0.2, 0.3],
    ],
    target="opx",
)

_apply_fem_output_port_delays(machine)

machine.save(QUAM_STATE_PATH)

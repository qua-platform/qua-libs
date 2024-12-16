# %%
import json
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import save_machine
import numpy as np


path = "./quam_state"

machine = QuAM.load()

u = unit(coerce_to_integer=True)

# Change active qubits
# machine.active_qubit_names = ["q0"]

# Update frequencies
rr_freq = np.array([8.55, 8.60, 8.65, 8.70]) * u.GHz
rr_LO = 8.75 * u.GHz
rr_if = rr_freq - rr_LO

xy_freq = np.array([6.20, 6.25, 6.30, 6.35]) * u.GHz
xy_LO = np.array([6.40, 6.40, 6.60, 6.60]) * u.GHz
xy_if_detuning = np.array([-5, -10, -15, -20]) * u.MHz
xy_if = rr_freq - rr_LO

machine.qubits["q1"].xy.thread = "a"
machine.qubits["q1"].resonator.thread = "a"

# NOTE: be aware of coupled ports for bands
for i, q in enumerate(machine.qubits):
    qb = machine.qubits[q]

    ## Update qubit rr freq and power
    qb.resonator.frequency_converter_up.LO_frequency = round(rr_LO)
    qb.resonator.frequency_converter_down.LO_frequency = round(rr_LO)
    qb.resonator.intermediate_frequency = round(rr_if[i])
    qb.resonator.thread = qb.name

    ## Update qubit xy freq and power
    qb.xy.frequency_converter_up.LO_frequency = round(xy_LO[i])
    qb.xy.intermediate_frequency = round(xy_if[i])
    qb.xy.thread = qb.name

    ## Update qubit xy detuned freq and power
    qb.xy_detuned.frequency_converter_up.LO_frequency = round(xy_LO[i])
    qb.xy_detuned.intermediate_frequency = qb.xy.intermediate_frequency + round(xy_if_detuning[i])
    qb.xy_detuned.thread = qb.name

    ## Update pulses
    # readout
    qb.resonator.operations["readout"].length = 2.5 * u.us
    qb.resonator.operations["readout"].amplitude = 1e-3
    # Qubit saturation
    qb.xy.operations["saturation"].length = 20 * u.us
    qb.xy.operations["saturation"].amplitude = 0.25
    # Single qubit gates - DragCosine
    qb.xy.operations["x180_DragCosine"].length = 48
    qb.xy.operations["x180_DragCosine"].amplitude = 0.2
    qb.xy.operations["x90_DragCosine"].amplitude = qb.xy.operations["x180_DragCosine"].amplitude / 2
    # Single qubit gates - Square
    qb.xy.operations["x180_Square"].length = 40
    qb.xy.operations["x180_Square"].amplitude = 0.1
    qb.xy.operations["x90_Square"].amplitude = qb.xy.operations["x180_Square"].amplitude / 2

from quam.components import pulses

for i, qp in enumerate(machine.qubit_pairs):
    qb_pair = machine.qubit_pairs[qp]
    qbc = qb_pair.qubit_control
    qbt = qb_pair.qubit_target
    cr = qb_pair.cross_resonance
    zz = qb_pair.zz_drive

    # CR gates - Square
    cr.thread = qbc.name
    qbt.xy.operations[f"{cr.name}_Square"] = pulses.SquarePulse(amplitude=-0.25, length=100, axis_angle=0, digital_marker="ON")
    qbt.xy.operations[f"{cr.name}_Square"].amplitude = 0.1
    qbt.xy.operations[f"{cr.name}_Square"].length = f"#/qubit_pairs/{qb_pair.name}/cross_resonance/operations/square/length"
    qbt.xy.operations[f"{cr.name}_Square"].axis_angle = 0  

    # ZZ gates - Square
    zz.thread = qbc.name
    zz.detuning = -15 * u.MHz
    # qbt.xy_detuned.detuning = f"#/qubit_pairs/{qb_pair.name}/zz_drive/intermediate_frquencys"
    qbt.xy_detuned.operations[f"{zz.name}_Square"] = pulses.SquarePulse(amplitude=-0.25, length=100, axis_angle=0, digital_marker="ON")
    qbt.xy_detuned.operations[f"{zz.name}_Square"].amplitude = 0.1
    qbt.xy_detuned.operations[f"{zz.name}_Square"].length = f"#/qubit_pairs/{qb_pair.name}/zz_drive/operations/square/length"
    qbt.xy_detuned.operations[f"{zz.name}_Square"].axis_angle = 0
        

# %%
# save into state.json
save_machine(machine, path)

# %%
# View the corresponding "raw-QUA" config
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

# %%

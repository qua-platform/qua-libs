# %%
import json
from qualang_tools.units import unit
from configuration.get_quam import QuAM
from quam_libs.quam_builder.superconducting.build_quam import save_machine
import numpy as np


path = r"C:\Users\KevinAVillegasRosale\OneDrive - QM Machines LTD\Documents\GitKraken\qua-libs\quam_state"

machine = QuAM.load(path)

u = unit(coerce_to_integer=True)

# Change active qubits
# machine.active_qubit_names = ["q0"]

for i in range(len(machine.qubits.items())):
    machine.qubits[f"q{i+1}"].grid_location = f"{i},0"

# Update frequencies
rr_freq = np.array([4.395, 4.412, 4.521]) * u.GHz
rr_LO = 4.75 * u.GHz
rr_if = rr_freq - rr_LO

xy_freq = np.array([6.012, 6.20, 6.4]) * u.GHz
# xy_LO = np.array([6.0, 6.5, 6.5]) * u.GHz
xy_LO = 6.3 * u.GHz
xy_if = xy_freq - xy_LO

# NOTE: be aware of coupled ports for bands
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    machine.qubits[q].resonator.intermediate_frequency = rr_if[i]

    ## Update qubit xy freq and power
    machine.qubits[q].xy.intermediate_frequency = xy_if[i]

    # Update flux channels
    machine.qubits[q].z.opx_output.output_mode = "amplified"
    machine.qubits[q].z.opx_output.upsampling_mode = "pulse"

    ## Update pulses
    # readout
    machine.qubits[q].resonator.operations["readout"].length = 2.5 * u.us
    machine.qubits[q].resonator.operations["readout"].amplitude = 1e-3
    # Qubit saturation
    machine.qubits[q].xy.operations["saturation"].length = 20 * u.us
    machine.qubits[q].xy.operations["saturation"].amplitude = 0.25
    # Single qubit gates - DragCosine
    machine.qubits[q].xy.operations["x180_DragCosine"].length = 48
    machine.qubits[q].xy.operations["x180_DragCosine"].amplitude = 0.2
    machine.qubits[q].xy.operations["x90_DragCosine"].amplitude = (
        machine.qubits[q].xy.operations["x180_DragCosine"].amplitude / 2
    )
    # Single qubit gates - Square
    machine.qubits[q].xy.operations["x180_Square"].length = 40
    machine.qubits[q].xy.operations["x180_Square"].amplitude = 0.1
    machine.qubits[q].xy.operations["x90_Square"].amplitude = (
        machine.qubits[q].xy.operations["x180_Square"].amplitude / 2
    )

# %%
# save into state.json
save_machine(machine, path)

# %%
# View the corresponding "raw-QUA" config
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

# %%

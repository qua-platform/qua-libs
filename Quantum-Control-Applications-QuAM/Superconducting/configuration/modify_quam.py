# %%
import json
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import save_machine
import numpy as np


def get_band(freq):
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} HZ is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


path = "./quam_state"

machine = QuAM.load(path)

u = unit(coerce_to_integer=True)

# Change active qubits
# machine.active_qubit_names = ["q0"]

for i in range(len(machine.qubits.items())):
    machine.qubits[f"q{i+1}"].grid_location = f"{i},0"

# Update frequencies
rr_freq = np.array([4.395, 4.412, 4.521, 4.728, 4.915, 5.147]) * u.GHz
rr_LO = 4.75 * u.GHz
rr_if = rr_freq - rr_LO
rr_max_power_dBm = 4

xy_freq = np.array([6.012, 6.421, 6.785, 7.001, 7.254, 5.978]) * u.GHz
xy_LO = np.array([6.0, 6.5, 6.5, 7.0, 7.04, 6.0]) * u.GHz
xy_if = xy_freq - xy_LO
xy_max_power_dBm = 1

# NOTE: be aware of coupled ports for bands
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    machine.qubits[q].resonator.opx_output.full_scale_power_dbm = rr_max_power_dBm
    machine.qubits[q].resonator.opx_output.upconverter_frequency = rr_LO
    machine.qubits[q].resonator.opx_input.downconverter_frequency = rr_LO
    machine.qubits[q].resonator.opx_input.band = get_band(rr_LO)
    machine.qubits[q].resonator.opx_output.band = get_band(rr_LO)
    machine.qubits[q].resonator.intermediate_frequency = rr_if[i]

    ## Update qubit xy freq and power
    machine.qubits[q].xy.opx_output.full_scale_power_dbm = xy_max_power_dBm
    machine.qubits[q].xy.opx_output.upconverter_frequency = xy_LO[i]
    machine.qubits[q].xy.opx_output.band = get_band(xy_LO[i])
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

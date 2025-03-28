########################################################################################################################
# %%                                             Import section
########################################################################################################################
import json
from qualang_tools.units import unit
from quam_config import QuAM
from quam_builder.builder.superconducting.build_quam import save_machine
from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses
from quam.components.pulses import GaussianPulse
import numpy as np


########################################################################################################################
# %%                                 QuAM loading and auxiliary functions
########################################################################################################################
# Loads the QuAM
machine = QuAM.load()
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


def get_band(freq):
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} HZ is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


########################################################################################################################
# %%                                    Gather the initial qubit parameters
########################################################################################################################
# Change active qubits
# machine.active_qubit_names = ["q0"]

for i in range(len(machine.qubits.items())):
    machine.qubits[f"q{i+1}"].grid_location = f"{i},0"

# Update frequencies
# NOTE: be aware of coupled ports for bands

# Resonator frequencies
rr_freq = np.array([4.395, 4.412, 4.521, 4.728, 4.915, 5.147, 5.247, 5.347]) * u.GHz
rr_LO = 4.75 * u.GHz
rr_if = rr_freq - rr_LO
rr_max_power_dBm = 4
# Qubit drive frequencies
xy_freq = np.array([6.012, 6.421, 6.785, 7.001, 7.083, 7.121, 7.184, 7.254]) * u.GHz
xy_LO = np.array([6.0, 6.1, 6.2, 6.3, 7.1, 7.1, 7.1, 7.1]) * u.GHz
xy_if = xy_freq - xy_LO
xy_max_power_dBm = 1


########################################################################################################################
# %%                             Initialize the QuAM with the initial qubit parameters
########################################################################################################################
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    machine.qubits[q].resonator.f_01 = rr_freq[i]
    machine.qubits[q].resonator.RF_frequency = machine.qubits[q].resonator.f_01
    machine.qubits[q].resonator.opx_output.full_scale_power_dbm = rr_max_power_dBm
    machine.qubits[q].resonator.opx_output.upconverter_frequency = rr_LO
    # machine.qubits[q].resonator.opx_input.downconverter_frequency = rr_LO
    machine.qubits[q].resonator.opx_input.band = get_band(rr_LO)
    machine.qubits[q].resonator.opx_output.band = get_band(rr_LO)

    ## Update qubit xy freq and power
    machine.qubits[q].f_01 = xy_freq[i]
    machine.qubits[q].xy.RF_frequency = machine.qubits[q].f_01
    machine.qubits[q].xy.opx_output.full_scale_power_dbm = xy_max_power_dBm
    machine.qubits[q].xy.opx_output.upconverter_frequency = xy_LO[i]
    machine.qubits[q].xy.opx_output.band = get_band(xy_LO[i])

    # Update flux channels
    machine.qubits[q].z.opx_output.output_mode = "direct"
    machine.qubits[q].z.opx_output.upsampling_mode = "pulse"

    ## Update pulses
    # readout
    machine.qubits[q].resonator.operations["readout"].length = 2.5 * u.us
    machine.qubits[q].resonator.operations["readout"].amplitude = 1e-3
    # Qubit saturation
    machine.qubits[q].xy.operations["saturation"].length = 20 * u.us
    machine.qubits[q].xy.operations["saturation"].amplitude = 0.25

    # Single qubit gates - DragCosine & Square
    add_DragCosine_pulses(machine.qubits[q], amplitude=0.25, length=48, alpha=0.0, detuning=0)
    # Single Gaussian flux pulse
    machine.qubits[q].z.operations["gauss"] = GaussianPulse(amplitude=0.1, length=200, sigma=40)

    # Add new pulses
    # from quam.components.pulses import (
    #     SquarePulse,
    #     DragGaussianPulse,
    #     DragCosinePulse,
    #     FlatTopGaussianPulse,
    #     WaveformPulse,
    #     SquareReadoutPulse,
    # )
    # e.g., machine.qubits[q].xy.operations["new_pulse"] = FlatTopGaussianPulse(...)

########################################################################################################################
# %%                                         Save the updated QuAM
########################################################################################################################
# save into state.json
save_machine(machine)

# %%
# View the corresponding "raw-QUA" config
# with open("qua_config.json", "w+") as f:
#     json.dump(machine.generate_config(), f, indent=4)

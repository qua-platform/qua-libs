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

########################################################################################################################
# %%                                    Gather the initial qubit parameters
########################################################################################################################
# Change active qubits
# machine.active_qubit_names = ["q0"]

for i in range(len(machine.qubits.items())):
    machine.qubits[f"q{i+1}"].grid_location = f"{i},0"

# Resonator frequencies
rr_freq = np.array([4.395, 4.412, 4.521, 4.785]) * u.GHz
rr_lo = 4.75 * u.GHz
rr_if = rr_freq - rr_lo
assert np.all(np.abs(rr_if) < 400 * u.MHz), "The resonator intermediate frequency must be within [-400; 400] MHz."
# Qubit drive frequencies
xy_freq = np.array([6.012, 6.20, 6.4, 6.79]) * u.GHz
xy_lo = np.array([6.0, 6.0, 6.5, 6.5]) * u.GHz
xy_if = xy_freq - xy_lo
assert np.all(np.abs(xy_if) < 400 * u.MHz), "The xy intermediate frequency must be within [-400; 400] MHz."

########################################################################################################################
# %%                             Initialize the QuAM with the initial qubit parameters
########################################################################################################################
# NOTE: be aware of coupled ports for bands
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    machine.qubits[q].resonator.f_01 = rr_freq[i]
    machine.qubits[q].resonator.RF_frequency = machine.qubits[q].resonator.f_01
    machine.qubits[q].resonator.frequency_converter_up.LO_frequency = rr_lo  # [2 : 0.250 : 18] GHz
    machine.qubits[q].resonator.frequency_converter_up.gain = 0  # [-20 : 0.5 : 20] dB
    machine.qubits[q].resonator.frequency_converter_up.output_mode = "always_on"  # "always_on" or "triggered"
    ## Update qubit xy freq and power
    machine.qubits[q].f_01 = xy_freq[i]
    machine.qubits[q].xy.RF_frequency = machine.qubits[q].f_01
    machine.qubits[q].xy.frequency_converter_up.LO_frequency = xy_lo[i]  # [2 : 0.250 : 18] GHz
    machine.qubits[q].xy.frequency_converter_up.gain = 0  # [-20 : 0.5 : 20] dB
    machine.qubits[q].xy.frequency_converter_up.output_mode = "always_on"  # "always_on" or "triggered"

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

########################################################################################################################
# %%                                         Save the updated QuAM
########################################################################################################################
machine.qubits[list(machine.qubits.to_dict().keys())[0]].print_summary()

config = machine.generate_config()
from pprint import pprint

pprint(machine.generate_config())
# save into state.json
save_machine(machine)

# %%
# View the corresponding "raw-QUA" config
# with open("qua_config.json", "w+") as f:
#     json.dump(machine.generate_config(), f, indent=4)

# %%

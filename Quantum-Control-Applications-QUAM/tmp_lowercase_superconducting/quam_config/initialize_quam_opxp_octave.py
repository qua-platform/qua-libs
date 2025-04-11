########################################################################################################################
# %%                                             Import section
########################################################################################################################
import json
from qualang_tools.units import unit
from quam_config import Quam
from quam_builder.builder.superconducting.build_quam import save_machine
from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses
from quam.components.pulses import GaussianPulse
import numpy as np
from pprint import pprint


########################################################################################################################
# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
# Loads the QUAM
machine = Quam.load()
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


def get_octave_gain_and_amplitude(desired_power: float, max_amplitude: float = 0.125) -> tuple[float, float]:
    """Get the Octave gain and IF amplitude for the Octave to output the specified desired power.

    :param desired_power: desired output power in dBm.
    :param max_amplitude: maximum allowed IF amplitude provided by the OPX to the Octave in V. Default is 0.125V,
    which is the optimum for the driving the Octave up-conversion mixers.
    :return: the Octave gain and IF amplitude realizing the desired power.
    """
    resulting_power = desired_power - u.volts2dBm(max_amplitude)
    if resulting_power < 0:
        octave_gain = round(max(desired_power - u.volts2dBm(max_amplitude) + 0.5, -20) * 2) / 2
    else:
        octave_gain = round(min(desired_power - u.volts2dBm(max_amplitude) + 0.5, 20) * 2) / 2
    amplitude = u.dBm2volts(desired_power - octave_gain)

    return octave_gain, amplitude


########################################################################################################################
# %%                                    Gather the initial qubit parameters
########################################################################################################################
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
# Transmon anharmonicity
anharmonicity = np.array([150, 200, 175, 310]) * u.MHz

# Desired output power in dBm
readout_power = -40
drive_power = -10

########################################################################################################################
# %%                             Initialize the QUAM with the initial qubit parameters
########################################################################################################################
# Get the Octave gain and IF amplitude corresponding to the desired powers
rr_gain, rr_amplitude = get_octave_gain_and_amplitude(readout_power, max_amplitude=0.125 / len(machine.qubits))
xy_gain, xy_amplitude = get_octave_gain_and_amplitude(drive_power)

# NOTE: be aware of coupled Octave channels and synthesizers
for i, q in enumerate(machine.qubits):
    ## Update qubit rr freq and power
    machine.qubits[q].resonator.f_01 = rr_freq[i]
    machine.qubits[q].resonator.RF_frequency = machine.qubits[q].resonator.f_01
    machine.qubits[q].resonator.frequency_converter_up.LO_frequency = rr_lo  # [2 : 0.250 : 18] GHz
    machine.qubits[q].resonator.frequency_converter_up.gain = rr_gain  # [-20 : 0.5 : 20] dB
    machine.qubits[q].resonator.frequency_converter_up.output_mode = "always_on"  # "always_on" or "triggered"
    ## Update qubit xy freq and power
    machine.qubits[q].f_01 = xy_freq[i]
    machine.qubits[q].xy.RF_frequency = machine.qubits[q].f_01
    machine.qubits[q].xy.frequency_converter_up.LO_frequency = xy_lo[i]  # [2 : 0.250 : 18] GHz
    machine.qubits[q].xy.frequency_converter_up.gain = xy_gain  # [-20 : 0.5 : 20] dB
    machine.qubits[q].xy.frequency_converter_up.output_mode = "always_on"  # "always_on" or "triggered"

    ## Update pulses
    # readout
    machine.qubits[q].resonator.operations["readout"].length = 2.5 * u.us
    machine.qubits[q].resonator.operations["readout"].amplitude = rr_amplitude
    # Qubit saturation
    machine.qubits[q].xy.operations["saturation"].length = 20 * u.us
    machine.qubits[q].xy.operations["saturation"].amplitude = 100 * u.mV

    # Single qubit gates - DragCosine & Square
    add_DragCosine_pulses(
        machine.qubits[q],
        amplitude=xy_amplitude,
        length=32,
        alpha=0.0,
        detuning=0,
        anharmonicity=anharmonicity.tolist()[i],
    )
    # Single Gaussian flux pulse
    if hasattr(machine.qubits[q], "z"):
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
# %%                                         Save the updated QUAM
########################################################################################################################
# save into state.json
save_machine(machine)

pprint(machine.generate_config())
# Save the corresponding "raw-QUA" config
# with open("qua_config.json", "w+") as f:
#     json.dump(machine.generate_config(), f, indent=4)

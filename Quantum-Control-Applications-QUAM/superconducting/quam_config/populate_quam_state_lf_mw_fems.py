########################################################################################################################
# %%                                             Import section
########################################################################################################################
import json
from qualang_tools.units import unit
from quam_config import Quam
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


def get_band(freq):
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} HZ is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


def closest_number(lst, target):
    return min(lst, key=lambda x: abs(x - target))


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.5) -> tuple[float, float]:
    """Get the Octave gain and IF amplitude for the Octave to output the specified desired power.

    :param desired_power: desired output power in dBm.
    :param max_amplitude: maximum allowed waveform amplitude in V. Default is 0.5V,
    :return: the full_scale_power_dBm and waveform amplitude realizing the desired power.
    """
    allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = closest_number(allowed_powers, max(resulting_power + 3, -11))
    else:
        full_scale_power_dBm = closest_number(allowed_powers, min(resulting_power + 3, 16))
    # amplitude =10**((desired_power - full_scale_power_dBm)/10)
    amplitude = 10 ** (-(full_scale_power_dBm - desired_power) / 20)
    return full_scale_power_dBm, amplitude


########################################################################################################################
# %%                                    Gather the initial qubit parameters
########################################################################################################################
for k, qubit in enumerate(machine.qubits.values()):
    qubit.grid_location = f"{k},0"

# Update frequencies
# NOTE: be aware of coupled ports for bands
# Resonator frequencies
rr_freq = np.array([4.395, 4.412, 4.521, 4.728, 4.915, 5.147, 5.247, 5.347]) * u.GHz
rr_LO = 4.75 * u.GHz
rr_if = rr_freq - rr_LO
# Qubit drive frequencies
xy_freq = np.array([6.012, 6.421, 6.785, 7.001, 7.083, 7.121, 7.184, 7.254]) * u.GHz
xy_LO = np.array([6.0, 6.1, 6.2, 6.3, 7.1, 7.1, 7.1, 7.1]) * u.GHz
xy_if = xy_freq - xy_LO
# Transmon anharmonicity
anharmonicity = np.array([150, 200, 175, 310]) * u.MHz

# Desired output power in dBm
readout_power = -40
drive_power = -10

########################################################################################################################
# %%                             Initialize the QUAM with the initial qubit parameters
########################################################################################################################
# Get the full_scale_power_dBm and waveform amplitude corresponding to the desired powers
rr_full_scale, rr_amplitude = get_full_scale_power_dBm_and_amplitude(
    readout_power, max_amplitude=0.125 / len(machine.qubits)
)
xy_full_scale, xy_amplitude = get_full_scale_power_dBm_and_amplitude(drive_power)

for k, qubit in enumerate(machine.qubits.values()):
    # Update qubit rr freq and power
    qubit.resonator.f_01 = rr_freq[k]
    qubit.resonator.RF_frequency = qubit.resonator.f_01
    qubit.resonator.opx_output.full_scale_power_dbm = rr_full_scale
    qubit.resonator.opx_output.upconverter_frequency = rr_LO
    qubit.resonator.opx_input.band = get_band(rr_LO)
    qubit.resonator.opx_output.band = get_band(rr_LO)

    # Update qubit xy freq and power
    qubit.f_01 = xy_freq[i]
    qubit.xy.RF_frequency = qubit.f_01
    qubit.xy.opx_output.full_scale_power_dbm = xy_full_scale
    qubit.xy.opx_output.upconverter_frequency = xy_LO[i]
    qubit.xy.opx_output.band = get_band(xy_LO[i])

    # Update flux channels
    qubit.z.opx_output.output_mode = "direct"
    qubit.z.opx_output.upsampling_mode = "pulse"
    # Update pulses
    # readout
    qubit.resonator.operations["readout"].length = 2.5 * u.us
    qubit.resonator.operations["readout"].amplitude = rr_amplitude
    # Qubit saturation
    qubit.xy.operations["saturation"].length = 20 * u.us
    qubit.xy.operations["saturation"].amplitude = 5 * xy_amplitude

    # Single qubit gates - DragCosine & Square
    add_DragCosine_pulses(qubit, amplitude=xy_amplitude, length=48, alpha=0.0, detuning=0)
    # Single Gaussian flux pulse
    if hasattr(qubit, "z"):
        qubit.z.operations["gauss"] = GaussianPulse(amplitude=0.1, length=200, sigma=40)

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
machine.save()

pprint(machine.generate_config())
# %%
# View the corresponding "raw-QUA" config
# with open("qua_config.json", "w+") as f:
#     json.dump(machine.generate_config(), f, indent=4)

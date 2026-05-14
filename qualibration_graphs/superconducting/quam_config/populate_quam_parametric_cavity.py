"""
Populate the parametric-cavity QuAM with initial device parameters.

Run ``generate_quam_parametric_cavity.py`` first to create state.json,
then run this script to fill in frequencies, powers, bands, and pulses.

Device
------
- Transmon (fixed frequency): 4.3 GHz
- Readout resonator:          4.5 GHz
- Storage cavity:             6.0 GHz
- Parametric coupling tones:  0.2, 1.5, 1.7 GHz  (all on MW-FEM O4, Band 1)
"""

########################################################################################################################
# %%                                             Import section
########################################################################################################################
import json
from pprint import pprint

import numpy as np
from qualang_tools.units import unit
from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses
from quam_config import Quam

from quam.components.pulses import SquarePulse, _FlatTopGaussianPulse

########################################################################################################################
# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
machine = Quam.load()
u = unit(coerce_to_integer=True)

if not machine.active_qubit_names:
    machine.active_qubit_names = list(machine.qubits.keys())


def get_band(freq):
    """Determine the MW-FEM DAC band for a given frequency."""
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(
            f"Frequency {freq} Hz is outside MW-FEM bandwidth [50 MHz, 10.5 GHz]"
        )


def closest_number(lst, target):
    return min(lst, key=lambda x: abs(x - target))


def get_full_scale_power_dBm_and_amplitude(
    desired_power: float, max_amplitude: float = 0.5
) -> tuple[int, float]:
    """Get full_scale_power_dbm and waveform amplitude for a desired output power.

    Args:
        desired_power: Desired output power in dBm.
        max_amplitude: Maximum allowed waveform amplitude (normalised).

    Returns:
        (full_scale_power_dBm, amplitude)
    """
    allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = closest_number(
            allowed_powers, max(resulting_power + 3, -11)
        )
    else:
        full_scale_power_dBm = closest_number(
            allowed_powers, min(resulting_power + 3, 16)
        )
    amplitude = 10 ** ((desired_power - full_scale_power_dBm) / 20)
    if -11 <= full_scale_power_dBm <= 16 and -1 <= amplitude <= 1:
        return full_scale_power_dBm, amplitude
    else:
        raise ValueError(
            f"Desired power outside specs ([-11;+16] dBm, [-1;+1]), "
            f"got ({full_scale_power_dBm}; {amplitude})"
        )


########################################################################################################################
# %%                                    Resonator parameters
########################################################################################################################
qubit = machine.qubits["q1"]

rr_freq = 4.5 * u.GHz
rr_upconv = 4.4 * u.GHz

readout_power = -40
rr_full_scale, rr_amplitude = get_full_scale_power_dBm_and_amplitude(
    readout_power, max_amplitude=0.125
)

qubit.resonator.f_01 = rr_freq
qubit.resonator.RF_frequency = rr_freq
qubit.resonator.opx_output.full_scale_power_dbm = rr_full_scale
qubit.resonator.opx_output.upconverter_frequency = rr_upconv
qubit.resonator.opx_output.band = get_band(rr_upconv)
qubit.resonator.opx_input.band = get_band(rr_upconv)

########################################################################################################################
# %%                                    Transmon XY parameters
########################################################################################################################
xy_freq = 4.3 * u.GHz
xy_upconv = 4.4 * u.GHz
anharmonicity_val = 200 * u.MHz

drive_power = -10
xy_full_scale, xy_amplitude = get_full_scale_power_dBm_and_amplitude(drive_power)

qubit.f_01 = xy_freq
qubit.xy.RF_frequency = xy_freq
qubit.xy.opx_output.full_scale_power_dbm = xy_full_scale
qubit.xy.opx_output.upconverter_frequency = xy_upconv
qubit.xy.opx_output.band = get_band(xy_upconv)
qubit.anharmonicity = anharmonicity_val
qubit.grid_location = "0,0"

########################################################################################################################
# %%                                    Cavity XY parameters
########################################################################################################################
cav_freq = 6.0 * u.GHz
cav_upconv = 6.0 * u.GHz

cav_drive_power = -20
cav_full_scale, cav_amplitude = get_full_scale_power_dBm_and_amplitude(cav_drive_power)

qubit.cavity.f_01 = cav_freq
qubit.cavity.xy.RF_frequency = cav_freq
qubit.cavity.xy.opx_output.full_scale_power_dbm = cav_full_scale
qubit.cavity.xy.opx_output.upconverter_frequency = cav_upconv
qubit.cavity.xy.opx_output.band = get_band(cav_upconv)

########################################################################################################################
# %%                                    Parametric drive parameters
########################################################################################################################
param_upconv = 1.6 * u.GHz

param_drive_power = -10
param_full_scale, param_amplitude = get_full_scale_power_dBm_and_amplitude(
    param_drive_power
)

qubit.parametric_drive.opx_output.full_scale_power_dbm = param_full_scale
qubit.parametric_drive.opx_output.upconverter_frequency = param_upconv
qubit.parametric_drive.opx_output.band = get_band(param_upconv)

qubit.parametric_drive.transmon_frequency = float(xy_freq)
qubit.parametric_drive.cavity_frequency = float(cav_freq)
qubit.parametric_drive.readout_frequency = float(rr_freq)

# Set the default RF frequency to the cavity-transmon coupling tone
cavity_transmon_detuning = abs(cav_freq - xy_freq)  # 1.7 GHz
qubit.parametric_drive.RF_frequency = float(cavity_transmon_detuning)

########################################################################################################################
# %%                                        Pulse parameters
########################################################################################################################
# --- Readout ---
qubit.resonator.operations["readout"].length = int(2.5 * u.us)
qubit.resonator.operations["readout"].amplitude = rr_amplitude
qubit.resonator.depletion_time = int(2.5 * u.us)

# --- Transmon saturation ---
qubit.xy.operations["saturation"].length = int(20 * u.us)
qubit.xy.operations["saturation"].amplitude = 0.1 * xy_amplitude

# --- Transmon single-qubit gates (DragCosine) ---
add_DragCosine_pulses(
    qubit,
    amplitude=xy_amplitude,
    length=40,
    anharmonicity=qubit.get_reference() + "/anharmonicity",
    alpha=0.0,
    detuning=0,
)

# --- Cavity saturation ---
qubit.cavity.xy.operations["saturation"].length = int(20 * u.us)
qubit.cavity.xy.operations["saturation"].amplitude = 0.1 * cav_amplitude

# --- Parametric coupling pulses ---
# Cavity-transmon SWAP (1.7 GHz)
qubit.parametric_drive.operations["cavity_transmon_swap"] = _FlatTopGaussianPulse(
    amplitude=param_amplitude,
    flat_length=200,
    smoothing_length=20,
    post_zero_padding_length=4,
)

# Cavity-readout coupling (1.5 GHz)
qubit.parametric_drive.operations["cavity_readout_coupling"] = _FlatTopGaussianPulse(
    amplitude=param_amplitude,
    flat_length=200,
    smoothing_length=20,
    post_zero_padding_length=4,
)

# Readout-transmon coupling (0.2 GHz)
qubit.parametric_drive.operations["readout_transmon_coupling"] = _FlatTopGaussianPulse(
    amplitude=param_amplitude,
    flat_length=200,
    smoothing_length=20,
    post_zero_padding_length=4,
)

########################################################################################################################
# %%                                         Save the updated QUAM
########################################################################################################################
machine.save()

pprint(machine.generate_config())
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)

print("QuAM populated and saved.")

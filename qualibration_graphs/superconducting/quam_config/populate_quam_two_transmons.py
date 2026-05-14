"""
Populate QUAM state for CS_4 two-qubit setup: q1 flux-tunable, q2 fixed-frequency (no flux line),
tunable coupler with z + xy, and CZ gate template macros on the pair.

Prerequisites: run ``generate_quam.py`` (with QUAM_AUTO_SAVE=1 for CI) so ``quam_state`` exists.
"""

########################################################################################################################
# %%                                             Import section
########################################################################################################################
import json
import os
from pprint import pprint

import numpy as np
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.custom_gates.flux_tunable_transmon_pair.two_qubit_gates import CZGate
from quam_builder.builder.superconducting.pulses import add_DragCosine_pulses
from quam.components.pulses import SquarePulse, _CosineBipolarPulse, _FlatTopGaussianPulse
from quam_config import Quam

########################################################################################################################
# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
machine = Quam.load()
u = unit(coerce_to_integer=True)


def get_band(freq):
    """MW-FEM DAC band for a given frequency (Hz)."""
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} Hz is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


def closest_number(lst, target):
    return min(lst, key=lambda x: abs(x - target))


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.5) -> tuple[int, float]:
    allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = closest_number(allowed_powers, max(resulting_power + 3, -11))
    else:
        full_scale_power_dBm = closest_number(allowed_powers, min(resulting_power + 3, 16))
    amplitude = 10 ** ((desired_power - full_scale_power_dBm) / 20)
    if -11 <= full_scale_power_dBm <= 16 and -1 <= amplitude <= 1:
        return full_scale_power_dBm, amplitude
    else:
        raise ValueError(
            f"The desired power is outside the specifications ([-11; +16]dBm, [-1; +1]), got ({full_scale_power_dBm}; {amplitude})"
        )


########################################################################################################################
# %%                                    Resonator parameters (q1, q2)
########################################################################################################################
rr_freq = np.array([5.5, 5.6]) * u.GHz
rr_LO = 5.5 * u.GHz
rr_if = rr_freq - rr_LO
assert np.all(np.abs(rr_if) < 400 * u.MHz), (
    "The resonator intermediate frequency must be within [-400; 400] MHz. \n"
    f"Readout frequencies: {rr_freq} \n"
    f"Readout LO frequency: {rr_LO} \n"
    f"Readout IF frequencies: {rr_if} \n"
)

readout_power = -40
rr_full_scale, rr_amplitude = get_full_scale_power_dBm_and_amplitude(
    readout_power, max_amplitude=0.125 / len(machine.qubits)
)

for k, qubit in enumerate(machine.qubits.values()):
    qubit.resonator.f_01 = rr_freq.tolist()[k]
    qubit.resonator.RF_frequency = qubit.resonator.f_01
    qubit.resonator.opx_output.full_scale_power_dbm = rr_full_scale
    qubit.resonator.opx_output.upconverter_frequency = rr_LO
    qubit.resonator.opx_input.band = get_band(rr_LO)
    qubit.resonator.opx_output.band = get_band(rr_LO)


########################################################################################################################
# %%                                    Qubit XY parameters
########################################################################################################################
# q1 > q2 so CZ template assigns flux-tunable q1 as control (has z line).
xy_freq = np.array([7.0, 6.5]) * u.GHz
xy_LO = np.array([7.0, 6.5]) * u.GHz
xy_if = xy_freq - xy_LO
assert np.all(np.abs(xy_if) < 400 * u.MHz), (
    "The xy intermediate frequency must be within [-400; 400] MHz. \n"
    f"Qubit drive frequencies: {xy_freq} \n"
    f"Qubit drive LO frequencies: {xy_LO} \n"
    f"Qubit drive IF frequencies: {xy_if} \n"
)
anharmonicity = np.array([-200, -200]) * u.MHz

drive_power = -10
xy_full_scale, xy_amplitude = get_full_scale_power_dBm_and_amplitude(drive_power)

for k, qubit in enumerate(machine.qubits.values()):
    qubit.f_01 = xy_freq.tolist()[k]
    qubit.xy.RF_frequency = qubit.f_01
    qubit.xy.opx_output.full_scale_power_dbm = xy_full_scale
    qubit.xy.opx_output.upconverter_frequency = xy_LO.tolist()[k]
    qubit.xy.opx_output.band = get_band(xy_LO.tolist()[k])
    qubit.grid_location = f"{k},0"
    qubit.anharmonicity = anharmonicity.tolist()[k]


########################################################################################################################
# %%                                    Flux parameters (q1 only)
########################################################################################################################
for qubit in machine.qubits.values():
    z = getattr(qubit, "z", None)
    if z is not None:
        z.opx_output.output_mode = "amplified"
        z.opx_output.upsampling_mode = "pulse"


########################################################################################################################
# %%                                    Tunable coupler (z + xy)
########################################################################################################################
coupler_xy_freq = 8.0 * u.GHz
coupler_xy_LO = 8.0 * u.GHz
assert abs(coupler_xy_freq - coupler_xy_LO) < 400 * u.MHz

for pair in machine.qubit_pairs.values():
    coupler = pair.coupler
    if coupler is None:
        continue
    if getattr(coupler, "z", None) is not None:
        coupler.z.opx_output.output_mode = "amplified"
        coupler.z.opx_output.upsampling_mode = "pulse"
    if getattr(coupler, "xy", None) is not None:
        cxy_lo = coupler_xy_LO
        coupler_xy_fsp, _ = get_full_scale_power_dBm_and_amplitude(-15)
        coupler.xy.opx_output.full_scale_power_dbm = coupler_xy_fsp
        coupler.xy.opx_output.upconverter_frequency = cxy_lo
        coupler.xy.opx_output.band = get_band(cxy_lo)
        coupler.xy.intermediate_frequency = coupler_xy_freq - cxy_lo
        coupler.xy.operations["cw"] = SquarePulse(length=1000, amplitude=0.1)


########################################################################################################################
# %%                                        Pulse parameters
########################################################################################################################
for q in machine.qubits:
    machine.qubits[q].resonator.operations["readout"].length = 2.5 * u.us
    machine.qubits[q].resonator.depletion_time = 2.5 * u.us
    machine.qubits[q].resonator.operations["readout"].amplitude = rr_amplitude
    machine.qubits[q].xy.operations["saturation"].length = 20 * u.us
    machine.qubits[q].xy.operations["saturation"].amplitude = 0.1 * xy_amplitude
    add_DragCosine_pulses(
        machine.qubits[q],
        amplitude=xy_amplitude,
        length=40,
        anharmonicity=machine.qubits[q].get_reference() + "/anharmonicity",
        alpha=0.0,
        detuning=0,
    )
    z = getattr(machine.qubits[q], "z", None)
    if z is not None:
        z.flux_point = "joint"


########################################################################################################################
# %%                                    Qubit pairs / CZ macros
########################################################################################################################
for pair in machine.qubit_pairs.values():
    cz_interaction_duration = 100
    smoothing_duration = 20
    post_zero_padding_length = 2

    print(f"Creating CZ Unipolar gate macro for {pair.name}")
    cz_pulse = SquarePulse(length=cz_interaction_duration, amplitude=0.1, id="cz_unipolar_pulse")
    cz = CZGate(flux_pulse_control=cz_pulse)
    pair.macros["cz_unipolar"] = cz
    pulse_length = pair.macros["cz_unipolar"].flux_pulse_control.get_reference() + "/length"
    pulse_amp = pair.macros["cz_unipolar"].flux_pulse_control.get_reference() + "/amplitude"
    pulse_name = pair.macros["cz_unipolar"].flux_pulse_control_label
    control_qb = pair.qubit_control
    control_qb.z.operations[pulse_name] = SquarePulse(length=cz_interaction_duration, amplitude=0.1)
    control_qb.z.operations[pulse_name].length = pulse_length
    control_qb.z.operations[pulse_name].amplitude = pulse_amp

    print(f"Creating CZ Flattop gate macro for {pair.name}")
    cz_pulse = _FlatTopGaussianPulse(
        amplitude=0.1,
        flat_length=cz_interaction_duration,
        smoothing_length=smoothing_duration,
        post_zero_padding_length=post_zero_padding_length,
        id="cz_flattop_pulse",
    )
    cz = CZGate(flux_pulse_control=cz_pulse)
    pair.macros["cz_flattop"] = cz
    flat_length = pair.macros["cz_flattop"].flux_pulse_control.get_reference() + "/flat_length"
    pulse_amp = pair.macros["cz_flattop"].flux_pulse_control.get_reference() + "/amplitude"
    pulse_smoothing = pair.macros["cz_flattop"].flux_pulse_control.get_reference() + "/smoothing_length"
    pulse_padding = pair.macros["cz_flattop"].flux_pulse_control.get_reference() + "/post_zero_padding_length"
    pulse_name = pair.macros["cz_flattop"].flux_pulse_control_label
    control_qb = pair.qubit_control
    control_qb.z.operations[pulse_name] = _FlatTopGaussianPulse(
        amplitude=0.1,
        flat_length=cz_interaction_duration,
        smoothing_length=smoothing_duration,
        post_zero_padding_length=post_zero_padding_length,
    )
    control_qb.z.operations[pulse_name].amplitude = pulse_amp
    control_qb.z.operations[pulse_name].flat_length = flat_length
    control_qb.z.operations[pulse_name].smoothing_length = pulse_smoothing
    control_qb.z.operations[pulse_name].post_zero_padding_length = pulse_padding

    print(f"Creating CZ Bipolar gate macro for {pair.name}")
    cz_pulse = _CosineBipolarPulse(
        amplitude=0.1,
        smoothing_length=smoothing_duration,
        id="cz_bipolar_pulse",
        flat_length=cz_interaction_duration,
        post_zero_padding_length=post_zero_padding_length,
    )
    cz = CZGate(flux_pulse_control=cz_pulse)
    pair.macros["cz_bipolar"] = cz
    flat_length = pair.macros["cz_bipolar"].flux_pulse_control.get_reference() + "/flat_length"
    pulse_amp = pair.macros["cz_bipolar"].flux_pulse_control.get_reference() + "/amplitude"
    pulse_smoothing = pair.macros["cz_bipolar"].flux_pulse_control.get_reference() + "/smoothing_length"
    pulse_padding = pair.macros["cz_bipolar"].flux_pulse_control.get_reference() + "/post_zero_padding_length"
    pulse_name = pair.macros["cz_bipolar"].flux_pulse_control_label
    control_qb = pair.qubit_control
    control_qb.z.operations[pulse_name] = _CosineBipolarPulse(
        amplitude=0.1,
        flat_length=cz_interaction_duration,
        smoothing_length=smoothing_duration,
        post_zero_padding_length=post_zero_padding_length,
    )
    control_qb.z.operations[pulse_name].amplitude = pulse_amp
    control_qb.z.operations[pulse_name].flat_length = flat_length
    control_qb.z.operations[pulse_name].smoothing_length = pulse_smoothing
    control_qb.z.operations[pulse_name].post_zero_padding_length = pulse_padding


########################################################################################################################
# %%                                         Save the updated QUAM
########################################################################################################################
machine.save()
cfg = machine.generate_config()
if os.environ.get("QUAM_PRINT_CONFIG", "").lower() in ("1", "true", "yes"):
    pprint(cfg)
with open("qua_config.json", "w+", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)
print(f"Wrote qua_config.json with {len(cfg.get('elements', {}))} elements.")

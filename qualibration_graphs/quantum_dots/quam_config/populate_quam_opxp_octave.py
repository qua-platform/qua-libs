"""Populate a QUAM state generated from ``wiring_opxp_octave.py``.

Run after ``generate_quam.py`` (with the OPX+ / Octave wiring example) to set
initial operational parameters before the first calibration.

Hardware assumptions
----------------------
- Sensor reflectometry: OPX+ input/output resonator line (``add_sensor_dot_resonator_line``;
  no Octave on readout).
- Qubit ESR drive: OPX+ baseband IQ → Octave upconversion (shared drive line).
"""

import numpy as np
from qualang_tools.units import unit

from quam_config import QubitQuam as Quam
from quam_builder.architecture.quantum_dots.operations.names import DrivePulseName, VoltagePointName

########################################################################################################################
# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
machine = Quam.load()
u = unit(coerce_to_integer=True)


def get_octave_gain_and_amplitude(desired_power: float, max_amplitude: float = 0.125) -> tuple[float, float]:
    """Return Octave gain [dB] and OPX IF amplitude [V] for a target output power."""
    resulting_power = desired_power - u.volts2dBm(max_amplitude)
    if resulting_power < 0:
        octave_gain = round(max(desired_power - u.volts2dBm(max_amplitude) + 0.5, -20) * 2) / 2
    else:
        octave_gain = round(min(desired_power - u.volts2dBm(max_amplitude) + 0.5, 20) * 2) / 2
    amplitude = u.dBm2volts(desired_power - octave_gain)

    if -20 <= octave_gain <= 20 and -0.5 <= amplitude < 0.5:
        return octave_gain, amplitude
    raise ValueError(
        f"The desired power is outside Octave specs ([-20; +20] dB, [-0.5; +0.5) V), "
        f"got ({octave_gain}; {amplitude})."
    )


#######################################
# %%   Qubits Physical Properties #####
#######################################
# Shared Octave LO for the multiplexed ESR drive line.
# Per-qubit ESR frequencies are applied as IF offsets in QUA via update_frequency.
xy_LO = 9.5 * u.GHz
xy_freq = np.array([9.1, 9.2, 9.523, 9.6]) * u.GHz
xy_if = xy_freq - xy_LO

drive_power = -10  # dBm at Octave RF output
xy_gain, xy_amplitude = get_octave_gain_and_amplitude(drive_power)

assert np.all(np.abs(xy_if) <= 400 * u.MHz), (
    "The xy intermediate frequency must be within [-400; 400] MHz.\n"
    f"Qubit drive frequencies: {xy_freq * 1e-9} GHz\n"
    f"Octave LO frequency: {xy_LO * 1e-9:.3f} GHz\n"
    f"Qubit drive IF frequencies: {xy_if * 1e-6} MHz\n"
)

#################################
# %%   Qubits Points Update #####
#################################
for i, q in enumerate(machine.qubits.values()):
    q.larmor_frequency = xy_freq[i]

    converter = getattr(q.xy, "frequency_converter_up", None)
    if converter is not None:
        converter.LO_frequency = xy_LO
        converter.gain = xy_gain
        converter.output_mode = "always_on"
    else:
        raise AttributeError(
            f"Qubit '{q.name}' xy drive has no frequency_converter_up — "
            "regenerate wiring with the OPX+ / Octave example."
        )

    for name in DrivePulseName:
        x90 = q.xy.operations.get(f"{name}_x90")
        x180 = q.xy.operations.get(f"{name}_x180")
        if x90 is not None:
            x90.amplitude = xy_amplitude / 2
        if x180 is not None:
            x180.amplitude = xy_amplitude

    q.T1 = 1e-6
    q.T2ramsey = 0.5e-6
    q.T2echo = 2e-6

    if q.name not in machine.active_qubit_names:
        machine.active_qubit_names.append(q.name)

for pair in machine.qubit_pairs.values():
    if pair.name not in machine.active_qubit_pair_names:
        machine.active_qubit_pair_names.append(pair.name)

##############################
# %%   Sensor Properties #####
##############################
resonator_frequencies = [250e6, 300e6]
readout_amplitude = 0.02  # In V
readout_length = 5 * u.us
# Define which quantum dot pair is used in the readout macro for reading out a given qubit
qubit_readout_dot_mapping = {
    "q1": "q2",
    "q2": "q1",
    "q3": "q4",
    "q4": "q3",
}

for i, sensor in enumerate(machine.sensor_dots.values()):
    resonator = sensor.readout_resonator
    resonator.intermediate_frequency = int(resonator_frequencies[i])
    resonator.opx_output.output_mode = "direct"
    resonator.opx_output.upsampling_mode = "mw"
    resonator.operations["readout"].amplitude = readout_amplitude
    resonator.operations["readout"].length = readout_length

for q in machine.qubits.values():
    q.preferred_readout_quantum_dot = qubit_readout_dot_mapping[q.name]

################################
# %%   Compensation Matrix #####
################################
qds = machine.quantum_dots
machine.update_cross_compensation_submatrix(
    virtual_names=["virtual_barrier_1", "virtual_barrier_2"],
    channels=[
        qds["virtual_dot_1"].physical_channel,
        qds["virtual_dot_2"].physical_channel,
        qds["virtual_dot_3"].physical_channel,
    ],
    matrix=[
        [0.1, 0.2],
        [0.3, 0.2],
        [0.2, 0.1],
    ],
    target="opx",
)
#########################
# %%   State Points #####
#########################
# ### Example generator method to add some default points. OPTIONAL

for i, qdp in enumerate(machine.quantum_dot_pairs.values()):
    qdp.add_point(
        point_name=VoltagePointName.INITIALIZE,
        voltages={d.id: (i + 1) * 0.015 for d in qdp.quantum_dots},
        duration=1000,
    )
    qdp.add_point(
        point_name=VoltagePointName.EMPTY,
        voltages={d.id: (i + 1) * 0.02 for d in qdp.quantum_dots},
        duration=1500,
    )
    qdp.add_point(
        point_name=VoltagePointName.MEASURE,
        voltages={d.id: (i + 1) * 0.025 for d in qdp.quantum_dots},
        duration=1000,
    )
    qdp.add_point(
        point_name=VoltagePointName.EXCHANGE,
        voltages={d.id: (i + 1) * -0.025 for d in qdp.quantum_dots},
        duration=1000,
    )


########################################################################################################################
# %%                                         Save the updated QUAM
########################################################################################################################
# save into state.json
machine.save()
# Visualize the QUA config and save it
# import json
# from pprint import pprint
# pprint(machine.generate_config())
# with open("qua_config.json", "w+") as f:
#     json.dump(machine.generate_config(), f, indent=4)

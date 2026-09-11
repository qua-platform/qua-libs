import numpy as np
from qualang_tools.units import unit
from quam_config import QubitQuam as Quam
from quam_builder.architecture.quantum_dots.operations.names import DrivePulseName, VoltagePointName

########################################################################################################################
# %%                                 QUAM loading and auxiliary functions
########################################################################################################################
# Loads the QUAM
machine = Quam.load()
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


def get_band(freq):
    """Determine the MW fem DAC band corresponding to a given frequency.

    Args:
        freq (float): The frequency in Hz.

    Returns:
        int: The Nyquist band number.
            - 1 if 50 MHz <= freq < 5.5 GHz
            - 2 if 4.5 GHz <= freq < 7.5 GHz
            - 3 if 6.5 GHz <= freq <= 10.5 GHz

    Raises:
        ValueError: If the frequency is outside the MW fem bandwidth [50 MHz, 10.5 GHz].
    """
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} Hz is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.99) -> tuple[int, float]:
    """Get the full_scale_power_dbm and waveform amplitude for the MW FEM to output the specified desired power.

    The keyword `full_scale_power_dbm` is the maximum power of normalized pulse waveforms in [-1,1].
    To convert to voltage:
        power_mw = 10**(full_scale_power_dbm / 10)
        max_voltage_amp = np.sqrt(2 * power_mw * 50 / 1000)
        amp_in_volts = waveform * max_voltage_amp
        ^ equivalent to OPX+ amp
    Its range is -11dBm to +18dBm with 1dBm steps.

    Args:
        desired_power (float): Desired output power in dBm.
        max_amplitude (float, optional): Maximum allowed waveform amplitude in V. Default is 0.5V.

    Returns:
        tuple[float, float]: The full_scale_power_dBm and waveform amplitude realizing the desired power.
    """
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = int(np.round(max(resulting_power, -11)))
    else:
        full_scale_power_dBm = int(np.round(min(resulting_power, 16)))
    amplitude = 10 ** ((desired_power - full_scale_power_dBm) / 20)
    if -11 <= full_scale_power_dBm <= 16 and -1 <= amplitude <= 1:
        return full_scale_power_dBm, amplitude
    else:
        raise ValueError(
            f"The desired power is outside the specifications ([-11; +16]dBm, [-1; +1]), got ({full_scale_power_dBm}; {amplitude})"
        )


def _apply_fem_output_port_delays(machine: Quam) -> None:
    """Set per-FEM analog output delays (LF path skew vs MW)."""
    for controller_ports in machine.ports.analog_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = LF_FEM_DELAY_NS

    for controller_ports in machine.ports.mw_outputs.values():
        for fem_ports in controller_ports.values():
            for port in fem_ports.values():
                port.delay = MW_FEM_DELAY_NS


# Align LF vs MW output timing (manual calibration might be necessary).
LF_FEM_DELAY_NS: int = 161
MW_FEM_DELAY_NS: int = 0
_apply_fem_output_port_delays(machine)


#######################################
# %%   Qubits Physical Properties #####
#######################################
# XY / MW-FEM: QuAM uses IF = larmor_frequency - MW_upconverter (see XYDriveMW).
# QM enforces |IF| <= 400 MHz.
larmor_center_hz = 9.5 * u.GHz  # The upconverter frequency
xy_freq = np.array([9.1, 9.2, 9.523, 9.6]) * u.GHz  # The qubit frequencies
xy_if = xy_freq - larmor_center_hz  # The inferred intermediate frequency

# Desired output power in dBm
drive_power = -10
# Get the full_scale_power_dBm and waveform amplitude corresponding to the desired powers
xy_full_scale, xy_amplitude = get_full_scale_power_dBm_and_amplitude(drive_power)

assert np.all(np.abs(xy_if) <= 400 * u.MHz), (
    "The xy intermediate frequency must be within [-400; 400] MHz. \n"
    f"Qubit drive frequencies: {xy_freq*1e-9} GHz\n"
    f"Qubit drive LO frequencies: {larmor_center_hz * 1e-9:.3f} GHz\n"
    f"Qubit drive IF frequencies: {xy_if * 1e-6} MHz\n"
)

#################################
# %%   Qubits Points Update #####
#################################
for i, q in enumerate(machine.qubits.values()):
    # MW FEM LO on this XY line (shared port → same value each iteration is fine).
    q.xy.opx_output.upconverter_frequency = larmor_center_hz
    q.larmor_frequency = xy_freq[i]
    q.xy.opx_output.band = get_band(xy_freq[i])  # Qubit drive band for the up-conversion
    q.xy.opx_output.full_scale_power_dbm = xy_full_scale  # Max drive power in dBm

    # Update all the existing pulse names based on enum DrivePulseName
    for name in DrivePulseName:
        # Ignore any pulses that are not mapped to the qubits (e.g. CROT, which is only mapped to the qubit_pair.)
        x90 = q.xy.operations.get(f"{name}_x90", None)
        x180 = q.xy.operations.get(f"{name}_x180", None)
        if x90 is not None:
            x90.amplitude = xy_amplitude / 2
        if x180 is not None:
            x180.amplitude = xy_amplitude

    # Default values in seconds
    q.T1 = 1e-6
    q.T2ramsey = 0.5e-6
    q.T2echo = 2e-6

    # Set all qubit to be active
    if q.name not in machine.active_qubit_names:
        machine.active_qubit_names.append(q.name)


##############################
# %%   Sensor Properties #####
##############################
resonator_frequencies = [250e6, 300e6]
readout_amplitude = 0.02  # In V
readout_length = 5 * u.us

for i, s in enumerate(machine.sensor_dots.values()):
    s.readout_resonator.intermediate_frequency = resonator_frequencies[i]
    s.readout_resonator.opx_output.output_mode = "direct"
    s.readout_resonator.opx_output.upsampling_mode = "mw"
    s.readout_resonator.operations["readout"].amplitude = readout_amplitude
    s.readout_resonator.operations["readout"].length = readout_length

################################
# %%   Compensation Matrix #####
################################

gate_set_id = next(iter(machine.virtual_gate_sets))

qds = machine.quantum_dots
# Orthogonalize the barriers. Detuning will be another layer.
machine.update_cross_compensation_submatrix(
    virtual_names=["virtual_barrier_1", "virtual_barrier_2", "virtual_barrier_3"],
    channels=[
        qds["virtual_dot_1"].physical_channel,
        qds["virtual_dot_2"].physical_channel,
        qds["virtual_dot_3"].physical_channel,
        qds["virtual_dot_4"].physical_channel,
    ],
    matrix=[
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
        [0.2, 0.1, 0.3],
        [0.1, 0.2, 0.3],
    ],
    target="opx",
)


#########################
# %%   State Points #####
#########################
# ### Example generator method to add some default points. OPTIONAL

# for i, qdp in enumerate(machine.quantum_dot_pairs.values()):
#     qdp.add_point(
#         point_name=VoltagePointName.INITIALIZE,
#         voltages={d.id: (i + 1) * 0.015 for d in qdp.quantum_dots},
#         duration=1000,
#     )
#     qdp.add_point(
#         point_name=VoltagePointName.EMPTY,
#         voltages={d.id: (i + 1) * 0.02 for d in qdp.quantum_dots},
#         duration=1500,
#     )
#     qdp.add_point(
#         point_name=VoltagePointName.MEASURE,
#         voltages={d.id: (i + 1) * 0.025 for d in qdp.quantum_dots},
#         duration=1000,
#     )
#     qdp.add_point(
#         point_name=VoltagePointName.EXCHANGE,
#         voltages={d.id: (i + 1) * -0.025 for d in qdp.quantum_dots},
#         duration=1000,
#     )


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

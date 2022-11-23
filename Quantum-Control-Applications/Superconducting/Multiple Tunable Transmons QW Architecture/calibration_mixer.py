"""
manual_mixer_calibration.py: Calibration for mixer imperfections
"""
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
from datetime import datetime
from time import sleep

##################
# State and QuAM #
##################
experiment = "mixer_calibration"
qubit_list = [0, 1]
qubit_index = 0
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

element = machine.qubits[qubit_index].name
# machine.readout_lines[qubit_index].lo_freq = 6.5e9
# machine.readout_resonators[qubit_index].f_res = 6.6457e9
machine.drive_lines[qubit_index].lo_freq = 7.5e9
machine.qubits[qubit_index].f_01 = 7.648e9

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
with program() as cw_output:
    with infinite_loop_():
        play("const" * amp(0), machine.qubits[qubit_index].name)


def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the 'I' & 'Q' ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the 'I' & 'Q' ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)
qm = qmm.open_qm(config)

job = qm.execute(cw_output)

# When done, the halt command can be called and the offsets can be written directly into the config file.

# job.halt()

# These are the 2 commands used to correct for mixer imperfections. The first is used to set the DC of the `I` and `Q`
# channels to compensate for the LO leakage. Since this compensation depends on the 'I' & 'Q' powers, it is advised to
# run this step with no input power, so that there is no LO leakage while the pulses are not played.
# The 2nd command is used to correct for the phase and amplitude mismatches between the channels.
# The output of the IQ Mixer should be connected to a spectrum analyzer and values should be chosen as to minimize the
# unwanted peaks.
# If python can read the output of the spectrum analyzer, then this process can be automated and the correct values can
# be found using an optimization method such as Nelder-Mead:
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html

# qm.set_output_dc_offset_by_element('qubit', ('I', 'Q'), (-0.001, 0.003))
# qm.set_mixer_correction('mixer_qubit', int(qubit_IF), int(qubit_LO), IQ_imbalance(0.015, 0.01))

# Automatic LO leakage correction
# centers = [0.5, 0]
# span = 0.1
#
# plt.figure()
# for n in range(3):
#     offset_i = np.linspace(centers[0] - span, centers[0] + span, 21)
#     offset_q = np.linspace(centers[1] - span, centers[1] + span, 31)
#     lo_leakage = np.zeros((len(offset_q), len(offset_i)))
#     for i in range(len(offset_i)):
#         for q in range(len(offset_q)):
#             qm.set_output_dc_offset_by_element(element, ("I", "Q"), (offset_i[i], offset_q[q]))
#             sleep(0.01)
#             # Write functions to extract the lo leakage from the spectrum analyzer
#             # lo_leakage[q][i] =
#     minimum = np.argwhere(lo_leakage == np.min(lo_leakage))[0]
#     centers = [offset_i[minimum[0]], offset_q[minimum[1]]]
#     span = span / 10
#     plt.subplot(131)
#     plt.pcolor(offset_i, offset_q, lo_leakage.transpose())
#     plt.xlabel("I offset [V]")
#     plt.ylabel("Q offset [V]")
#     plt.title(
#         f"Minimum at (I={centers[0]:.3f}, Q={centers[1]:.3f}) = {lo_leakage[minimum[0]][minimum[1]]:.1f} dBm"
#     )
# plt.suptitle(f"LO leakage correction for {element}")

# Automatic image cancellation
# centers = [0.5, 0]
# span = [0.2, 0.5]
#
# plt.figure()
# for n in range(3):
#     gain = np.linspace(centers[0] - span, centers[0] + span, 21)
#     phase = np.linspace(centers[1] - span, centers[1] + span, 31)
#     image = np.zeros((len(phase), len(gain)))
#     for g in range(len(gain)):
#         for p in range(len(phase)):
#             qm.set_mixer_correction(
#                 config["elements"][element]["mixInputs"]["mixer"],
#                 int(config["elements"][element]["intermediate_frequency"]),
#                 int(config["elements"][element]["mixInputs"]["lo_frequency"]),
#                 IQ_imbalance(gain[g], phase[p]),
#             )
#             sleep(0.01)
#             # Write functions to extract the image from the spectrum analyzer
#             # image[q][i] =
#     minimum = np.argwhere(image == np.min(image))[0]
#     centers = [gain[minimum[0]], phase[minimum[1]]]
#     span = (np.array(span) / 10).tolist()
#     plt.subplot(131)
#     plt.pcolor(gain, phase, image.transpose())
#     plt.xlabel("Gain")
#     plt.ylabel("Phase imbalance [rad]")
#     plt.title(
#         f"Minimum at (I={gain[centers[0]]:.3f}, Q={phase[centers[1]]:.3f}) = {image[centers[0]][centers[1]]:.1f} dBm"
#     )
# plt.suptitle(f"Image cancellation for {element}")

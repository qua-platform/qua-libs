"""
        ACTIVE RESET
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated the IQ blobs (rotation_angle and ge_threshold).
    - (optional) Having calibrated the readout (readout_frequency_, _amplitude_, _duration_optimization).
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the configuration.
    - Update the g -> e threshold (ge_threshold) in the configuration.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from qualang_tools.analysis.discriminator import two_state_discriminator

import matplotlib.pyplot as plt

##############################
# Program-specific variables #
##############################
threshold = ge_threshold  # Threshold used for ge state discrimination
n_shot = 10000  # Number of acquired shots
max_tries = 2  # Maximum number of tries for active reset (no feedback if set to 0)


def qubit_initialization():
    assign(count, 0)
    assign(cont_condition, ((I > threshold) & (count < max_tries)))
    with while_(cont_condition):
        play("x180", "qubit")
        align("qubit", "resonator")
        measure("readout", "resonator", None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
        assign(count, count + 1)
        assign(cont_condition, ((I > threshold) & (count < max_tries)))
    return count

def active_reset_one_threshold(threshold_g, max_tries):
    I_reset = declare(fixed)
    counter = declare(int)
    assign(counter, 0)
    align("resonator", "qubit")
    while (I_reset > threshold_g) & (counter < max_tries):
        measure("readout", "resonator", None, dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_reset))
        align("resonator", "qubit")
        wait(depletion_time * u.ns, "qubit")
        play("x180", "qubit", condition=(I_reset > threshold_g))
        assign(counter, counter + 1)
    return count
def active_reset_two_thresholds(threshold_g, threshold_e, max_tries):
    I_reset = declare(fixed)
    counter = declare(int)
    assign(counter, 0)
    align("resonator", "qubit")
    while (I_reset > threshold_g) & (counter < max_tries):
        measure("readout", "resonator", None, dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_reset))
        align("resonator", "qubit")
        wait(depletion_time * u.ns, "qubit")
        play("x180", "qubit", condition=(I_reset > threshold_e))
        assign(counter, counter + 1)
    return count
def active_reset_fast(threshold_g):
    I_reset = declare(fixed)
    align("resonator", "qubit")
    measure("readout", "resonator", None, dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_reset))
    align("resonator", "qubit")
    wait(depletion_time * u.ns, "qubit")
    play("x180", "qubit", condition=(I > threshold_g))

###################
# The QUA program #
###################

with program() as active_reset_prog:
    n = declare(int)  # Averaging index
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    count = declare(int)
    cont_condition = declare(bool)
    tries_st = declare_stream()

    with for_(n, 0, n < n_shot, n + 1):
        # Active reset
        count = qubit_initialization()
        align()
        # Measure the state of the resonator
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
        )
        # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
        save(I_g, I_g_st)
        save(Q_g, Q_g_st)
        with if_(count > 0):
            save(count, tries_st)

        align()  # global align
        # Active reset
        count = qubit_initialization()
        align()
        # Play the x180 gate to put the qubit in the excited state
        play("x180", "qubit")
        # Align the two elements to measure after playing the qubit pulse.
        align("qubit", "resonator")
        # Measure the state of the resonator
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
        )
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I_e, I_e_st)
        save(Q_e, Q_e_st)
        with if_(count > 0):
            save(count, tries_st)

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")
        tries_st.average().save("average_tries")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, active_reset_prog, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(active_reset_prog)
    # Creates a result handle to fetch data from the OPX
    res_handles = job.result_handles
    # Waits (blocks the Python console) until all results have been acquired
    res_handles.wait_for_all_values()
    # Fetch the 'I' & 'Q' points for the qubit in the ground and excited states
    Ig = res_handles.get("I_g").fetch_all()["value"]
    Qg = res_handles.get("Q_g").fetch_all()["value"]
    Ie = res_handles.get("I_e").fetch_all()["value"]
    Qe = res_handles.get("Q_e").fetch_all()["value"]
    average_tries = res_handles.get("average_tries").fetch_all()["value"]
    # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
    # for state discrimination and derive the fidelity matrix
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)
    plt.suptitle(f"{average_tries=}")
    print(f"{average_tries=}")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

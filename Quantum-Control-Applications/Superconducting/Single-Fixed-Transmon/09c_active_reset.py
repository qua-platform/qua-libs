"""
        ACTIVE RESET
This script is used to benchmark different types of qubit initialization including active reset protocols.
the different methods are written in macros for better readability.

Each protocol is detailed in the corresponding docstring, but the idea behind active reset is to first measure one
quadrature of the resonator ("I") and compare it to one or two threshold in order to decide whether to apply a pi-pulse
(qubit in |e>), do nothing (qubit in |g>) or measure again if the qubit state is undetermined (active_reset_two_thresholds).

Then, after qubit initialization, the IQ blobs for |g> and |e> are measured again and the readout fidelity is derived
similarly to what is done in IQ_blobs.py.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated the IQ blobs (rotation_angle and ge_threshold).
    - (optional) Having calibrated the readout (readout_frequency_, _amplitude_, _duration_optimization).
    - Having updated the rotation angle (rotation_angle) and g -> e threshold (ge_threshold) in the configuration (IQ_blobs.py).
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from qualang_tools.analysis.discriminator import two_state_discriminator

import matplotlib.pyplot as plt

##############################
# Program-specific variables #
##############################
# "thermalization", "active_reset_one_threshold", "active_reset_two_thresholds", "active_reset_fast"
initialization_method = "active_reset_one_threshold"
n_shot = 10000  # Number of acquired shots
# The thresholds ar calibrated with the IQ_blobs.py script:
# If I > threshold_e, then the qubit is assumed to be in |e> and a pi pulse is played to reset it.
# If I < threshold_g, then the qubit is assumed to be in |g>.
# else, the qubit state is not determined accurately enough, so we just measure again.
ge_threshold_g = ge_threshold * 0.5
ge_threshold_e = ge_threshold
# Maximum number of tries for active reset
max_tries = 2


def qubit_initialization(method: str = "thermalization"):
    """
    Allows to switch between several initialization methods.

    :param method: the desired initialization method among "thermalization", "active_reset_one_threshold", "active_reset_two_thresholds", "active_reset_fast".
    :return: the number of tries to reset the qubit.
    """
    if method == "thermalization":
        wait(thermalization_time * u.ns)
        return 1
    elif method == "active_reset_fast":
        return active_reset_fast(ge_threshold_e)
    elif method == "active_reset_one_threshold":
        return active_reset_one_threshold(ge_threshold_e, max_tries)
    elif method == "active_reset_two_thresholds":
        return active_reset_two_thresholds(ge_threshold_g, ge_threshold_e, max_tries)
    else:
        raise ValueError(f"method {method} is not implemented.")


def active_reset_one_threshold(threshold_g: float, max_tries: int):
    """
    Active reset protocol where the outcome of the measurement is compared to a pre-calibrated threshold (IQ_blobs.py).
    If the qubit is in |e> (I>threshold), then play a pi pulse and measure again, else (qubit in |g>) return the number
    of pi-pulses needed to reset the qubit.
    The program waits for the resonator to deplete before playing the conditional pi-pulse so that the calibrated
    pi-pulse parameters are still valid.

    :param threshold_g: threshold between the |g> and |e> blobs - calibrated in IQ_blobs.py
    :param max_tries: maximum number of iterations needed to reset the qubit before exiting the loop anyway.
    :return: the number of tries to reset the qubit.
    """
    I_reset = declare(fixed)
    counter = declare(int)
    assign(counter, 0)
    align("resonator", "qubit")
    with while_((I_reset > threshold_g) & (counter < max_tries)):
        # Measure the state of the resonator
        measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_reset))
        align("resonator", "qubit")
        # Wait for the resonator to deplete
        wait(depletion_time * u.ns, "qubit")
        # Play a conditional pi-pulse to actively reset the qubit
        play("x180", "qubit", condition=(I_reset > threshold_g))
        # Update the counter for benchmarking purposes
        assign(counter, counter + 1)
    return counter


def active_reset_two_thresholds(threshold_g: float, threshold_e: float, max_tries: int):
    """
    Active reset protocol where the outcome of the measurement is compared to two pre-calibrated thresholds (IQ_blobs.py).
    If I > threshold_e, then the qubit is assumed to be in |e> and a pi pulse is played to reset it.
    If I < threshold_g, then the qubit is assumed to be in |g> and the loop can be exited.
    else, the qubit state is not determined accurately enough, so we just repeat the process.
    The program waits for the resonator to deplete before playing the conditional pi-pulse so that the calibrated
    pi-pulse parameters are still valid.

    :param threshold_g: threshold "inside" the |g> blob, below which the qubit is in |g> with great certainty.
    :param threshold_e: threshold between the |g> and |e> blobs - calibrated in IQ_blobs.py
    :param max_tries: maximum number of iterations needed to reset the qubit before exiting the loop anyway.
    :return: the number of tries to reset the qubit.
    """
    I_reset = declare(fixed)
    counter = declare(int)
    assign(counter, 0)
    align("resonator", "qubit")
    with while_((I_reset > threshold_g) & (counter < max_tries)):
        # Measure the state of the resonator
        measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_reset))
        align("resonator", "qubit")
        # Wait for the resonator to deplete
        wait(depletion_time * u.ns, "qubit")
        # Play a conditional pi-pulse to actively reset the qubit
        play("x180", "qubit", condition=(I_reset > threshold_e))
        # Update the counter for benchmarking purposes
        assign(counter, counter + 1)
    return counter


def active_reset_fast(threshold_g: float):
    """
    Active reset protocol where the outcome of the measurement is compared to a pre-calibrated threshold (IQ_blobs.py).
    If the qubit is in |e> (I>threshold), then play a pi pulse, else (qubit in |g>) do nothing and proceed to the sequence.
    The program waits for the resonator to deplete before playing the conditional pi-pulse so that the calibrated
    pi-pulse parameters are still valid.

    :param threshold_g: threshold between the |g> and |e> blobs - calibrated in IQ_blobs.py.
    :return: 1
    """
    I_reset = declare(fixed)
    align("resonator", "qubit")
    # Measure the state of the resonator
    measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_reset))
    align("resonator", "qubit")
    # Wait for the resonator to deplete
    wait(depletion_time * u.ns, "qubit")
    # Play a conditional pi-pulse to actively reset the qubit
    play("x180", "qubit", condition=(I_reset > threshold_g))
    return 1


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

    cont_condition = declare(bool)
    tries_st = declare_stream()

    with for_(n, 0, n < n_shot, n + 1):
        # Active reset
        count = qubit_initialization(method=initialization_method)
        align()
        # Measure the state of the resonator after reset, qubit should be in |g>
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_g),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_g),
        )
        # Save the 'I' & 'Q' quadratures to their respective streams for the ground state
        save(I_g, I_g_st)
        save(Q_g, Q_g_st)
        with if_(count > 0):
            save(count, tries_st)

        align()  # global align
        # Active reset
        count = qubit_initialization(method=initialization_method)
        align()
        # Play the x180 gate to put the qubit in the excited state
        play("x180", "qubit")
        # Align the two elements to measure after playing the qubit pulse.
        align("qubit", "resonator")
        # Measure the state of the resonator, qubit should be in |e>
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "rotated_sin", I_e),
            dual_demod.full("rotated_minus_sin", "rotated_cos", Q_e),
        )
        # Save the 'I' & 'Q' quadratures to their respective streams for the excited state
        save(I_e, I_e_st)
        save(Q_e, Q_e_st)
        # Save only the count when the qubit was not directly measured in |g>
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
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

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
    average_tries = res_handles.get("average_tries").fetch_all()
    # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
    # for state discrimination and derive the fidelity matrix
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)
    plt.suptitle(f"{average_tries=}")
    print(f"{average_tries=}")

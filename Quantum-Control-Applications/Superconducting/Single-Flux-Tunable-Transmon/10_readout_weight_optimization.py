"""
        READOUT OPTIMISATION: INTEGRATION WEIGHTS
This sequence involves assessing the state of the resonator in two distinct scenarios: first, after thermalization
(with the qubit in the |g> state) and then following the application of a pi pulse to the qubit (transitioning the
qubit to the |e> state).
The "demod.sliced" method is employed to capture the time trace of the demodulated data, providing insight into the
resonator's response.
Reference: https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=accumulated#sliced-demodulation

From the average I & Q quadratures for the qubit states |g> and |e>, along with their variances,
the Signal-to-Noise Ratio (SNR) is determined. The readout duration that yields the highest SNR is selected as
the optimal choice.
It's important to note that if you aim to observe the resonator's behavior during its ringdown phase,
the length of the integration weights should surpass that of the readout_pulse.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated the readout frequency, amplitude and duration and updated the configuration.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the integration weights in the configuration by adding
    config["integration_weights"]["opt_cos_weights"] = {"cosine": weights_cos, "sine": weights_minus_sin}
    config["integration_weights"]["opt_sin_weights"] = {"cosine": weights_sin, "sine": weights_cos}
    config["integration_weights"]["opt_minus_sin_weights"] = {"cosine": weights_minus_sin, "sine": weights_minus_cos}
    # also need to add the new weights to readout_pulse
    config['pulses']['readout_pulse']['integration_weights'] = ['opt_cos', 'opt_sin', 'opt_minus_sin']
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
import warnings

warnings.filterwarnings("ignore")


###########
# Helpers #
###########
def divide_array_in_half(arr):
    split_index = len(arr) // 2
    arr1 = arr[:split_index]
    arr2 = arr[split_index:]
    return arr1, arr2


def normalize_complex_array(arr):
    # Calculate the simple norm of the complex array
    norm = np.sqrt(np.sum(np.abs(arr) ** 2))

    # Normalize the complex array by dividing it by the norm
    normalized_arr = arr / norm

    # Rescale the normalized array so that the maximum value is 1
    max_val = np.max(np.abs(normalized_arr))
    rescaled_arr = normalized_arr / max_val

    return rescaled_arr


def plot_three_complex_arrays(arr1, arr2, arr3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(arr1.real, label="real")
    ax1.plot(arr1.imag, label="imag")
    ax1.set_title("ground state")
    ax1.set_xlabel("Clock cycles")
    ax1.set_ylabel("demod traces [a.u.]")
    ax1.legend()
    ax2.plot(arr2.real, label="real")
    ax2.plot(arr2.imag, label="imag")
    ax2.set_title("excited state")
    ax2.set_xlabel("Clock cycles")
    ax2.set_ylabel("demod traces [a.u.]")
    ax2.legend()
    ax3.plot(arr3.real, label="real")
    ax3.plot(arr3.imag, label="imag")
    ax3.set_title("SNR")
    ax3.set_xlabel("Clock cycles")
    ax3.set_ylabel("subtracted traces [a.u.]")
    ax3.legend()
    plt.tight_layout()
    plt.show()


###################
# The QUA program #
###################
division_length = 1  # Size of each slice in clock cycles
number_of_divisions = int(readout_len / (4 * division_length))  # Number of slices
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

n_avg = 100  # number of averages


with program() as opt_weights:
    n = declare(int)
    ind = declare(int)
    II = declare(fixed, size=number_of_divisions)
    IQ = declare(fixed, size=number_of_divisions)
    QI = declare(fixed, size=number_of_divisions)
    QQ = declare(fixed, size=number_of_divisions)

    n_st = declare_stream()
    II_st = declare_stream()
    IQ_st = declare_stream()
    QI_st = declare_stream()
    QQ_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Measure the ground state
        measure(
            "readout",
            "resonator",
            None,
            demod.sliced("cos", II, division_length, "out1"),
            demod.sliced("sin", IQ, division_length, "out2"),
            demod.sliced("minus_sin", QI, division_length, "out1"),
            demod.sliced("cos", QQ, division_length, "out2"),
        )
        wait(thermalization_time * u.ns, "resonator")
        # Save the sliced data (time trace of the demodulated data with a resolution equals to the division length)
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(II[ind], II_st)
            save(IQ[ind], IQ_st)
            save(QI[ind], QI_st)
            save(QQ[ind], QQ_st)
        wait(thermalization_time * u.ns, "resonator")

        align()  # Global align to play the pi pulse after thermalization

        # Measure the excited state
        play("x180", "qubit")
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            demod.sliced("cos", II, division_length, "out1"),
            demod.sliced("sin", IQ, division_length, "out2"),
            demod.sliced("minus_sin", QI, division_length, "out1"),
            demod.sliced("cos", QQ, division_length, "out2"),
        )
        wait(thermalization_time * u.ns, "resonator")
        # Save the sliced data (time trace of the demodulated data with a resolution equals to the division length)
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(II[ind], II_st)
            save(IQ[ind], IQ_st)
            save(QI[ind], QI_st)
            save(QQ[ind], QQ_st)
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        II_st.buffer(2 * number_of_divisions).average().save("II")
        IQ_st.buffer(2 * number_of_divisions).average().save("IQ")
        QI_st.buffer(2 * number_of_divisions).average().save("QI")
        QQ_st.buffer(2 * number_of_divisions).average().save("QQ")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, opt_weights, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(opt_weights)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()[0]
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # Fetch and reshape the data
    res_handles = job.result_handles
    IIe, IIg = divide_array_in_half(res_handles.get("II").fetch_all())
    IQe, IQg = divide_array_in_half(res_handles.get("IQ").fetch_all())
    QIe, QIg = divide_array_in_half(res_handles.get("QI").fetch_all())
    QQe, QQg = divide_array_in_half(res_handles.get("QQ").fetch_all())
    # Sum the quadrature to fully demodulate the traces
    Ie = IIe + IQe
    Ig = IIg + IQg
    Qe = QIe + QQe
    Qg = QIg + QQg
    # Derive and normalize the ground and excited traces
    ground_trace = Ig + 1j * Qg
    excited_trace = Ie + 1j * Qe
    subtracted_trace = excited_trace - ground_trace
    norm_subtracted_trace = normalize_complex_array(subtracted_trace)  # <- these are the optimal weights :)
    # Plot the results
    plot_three_complex_arrays(ground_trace, excited_trace, norm_subtracted_trace)
    # Reshape the optimal integration weights to match the configuration
    from qualang_tools.config.integration_weights_tools import convert_integration_weights

    weights_cos = convert_integration_weights(list(norm_subtracted_trace.real))
    weights_minus_sin = convert_integration_weights(list((-1) * norm_subtracted_trace.imag))
    weights_sin = convert_integration_weights(list(norm_subtracted_trace.imag))
    weights_minus_cos = convert_integration_weights(list((-1) * norm_subtracted_trace.real))
    # After obtaining the optimal weights, you need to load them to the 'integration_weights' dictionary in the config
    # config["integration_weights"]["opt_cos_weights"] = {"cosine": weights_cos, "sine": weights_minus_sin}
    # config["integration_weights"]["opt_sin_weights"] = {"cosine": weights_sin, "sine": weights_cos}
    # config["integration_weights"]["opt_minus_sin_weights"] = {"cosine": weights_minus_sin, "sine": weights_minus_cos}
    # also need to add the new weights to readout_pulse
    # config['pulses']['readout_pulse']['integration_weights'] = ['opt_cos', 'opt_sin', 'opt_minus_sin']
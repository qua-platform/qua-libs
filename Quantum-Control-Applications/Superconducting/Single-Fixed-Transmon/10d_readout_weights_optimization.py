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

Next steps before going to the next node:
    - Update the integration weights in the configuration by following the steps at the end of the script.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
import matplotlib.pyplot as plt


####################
# Helper functions #
####################
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


def plot_three_complex_arrays(x, arr1, arr2, arr3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(x, arr1.real, label="real")
    ax1.plot(x, arr1.imag, label="imag")
    ax1.set_title("ground state")
    ax1.set_xlabel("Readout time [ns]")
    ax1.set_ylabel("demod traces [a.u.]")
    ax1.legend()
    ax2.plot(x, arr2.real, label="real")
    ax2.plot(x, arr2.imag, label="imag")
    ax2.set_title("excited state")
    ax2.set_xlabel("Readout time [ns]")
    ax2.set_ylabel("demod traces [a.u.]")
    ax2.legend()
    ax3.plot(x, arr3.real, label="real")
    ax3.plot(x, arr3.imag, label="imag")
    ax3.set_title("SNR")
    ax3.set_xlabel("Readout time [ns]")
    ax3.set_ylabel("subtracted traces [a.u.]")
    ax3.legend()
    plt.tight_layout()
    plt.show()


def update_readout_length(new_readout_length, ringdown_length):
    config["pulses"]["readout_pulse"]["length"] = new_readout_length
    config["integration_weights"]["cosine_weights"] = {
        "cosine": [(1.0, new_readout_length + ringdown_length)],
        "sine": [(0.0, new_readout_length + ringdown_length)],
    }
    config["integration_weights"]["sine_weights"] = {
        "cosine": [(0.0, new_readout_length + ringdown_length)],
        "sine": [(1.0, new_readout_length + ringdown_length)],
    }
    config["integration_weights"]["minus_sine_weights"] = {
        "cosine": [(0.0, new_readout_length + ringdown_length)],
        "sine": [(-1.0, new_readout_length + ringdown_length)],
    }


###################
# The QUA program #
###################
n_avg = 100  # number of averages
# Set maximum readout duration for this scan and update the configuration accordingly
readout_len = 5 * u.us  # Readout pulse duration
ringdown_len = 0 * u.us  # integration time after readout pulse to observe the ringdown of the resonator
update_readout_length(readout_len, ringdown_len)
# Set the sliced demod parameters
division_length = 10  # Size of each demodulation slice in clock cycles
number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))  # Number of slices
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)

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
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

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
    IIg, IIe = divide_array_in_half(res_handles.get("II").fetch_all())
    IQg, IQe = divide_array_in_half(res_handles.get("IQ").fetch_all())
    QIg, QIe = divide_array_in_half(res_handles.get("QI").fetch_all())
    QQg, QQe = divide_array_in_half(res_handles.get("QQ").fetch_all())
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
    plot_three_complex_arrays(x_plot, ground_trace, excited_trace, norm_subtracted_trace)
    # Reshape the optimal integration weights to match the configuration
    weights_real = norm_subtracted_trace.real
    weights_minus_imag = -norm_subtracted_trace.imag
    weights_imag = norm_subtracted_trace.imag
    weights_minus_real = -norm_subtracted_trace.real
    # Save the weights for later use in the config
    np.savez(
        "optimal_weights",
        weights_real=weights_real,
        weights_minus_imag=weights_minus_imag,
        weights_imag=weights_imag,
        weights_minus_real=weights_minus_real,
        division_length=division_length,
    )
    # After obtaining the optimal weights, you need to load them to the 'integration_weights' dictionary in the config.
    # For this, you can just copy and paste the following lines into the "integration_weights" section:
    # "opt_cosine_weights": {
    #     "cosine": opt_weights_real,
    #     "sine": opt_weights_minus_imag,
    # },
    # "opt_sine_weights": {
    #     "cosine": opt_weights_imag,
    #     "sine": opt_weights_real,
    # },
    # "opt_minus_sine_weights": {
    #     "cosine": opt_weights_minus_imag,
    #     "sine": opt_weights_minus_real,
    # },

    # also need to add the new weights to readout_pulse under the "integration_weights" section:
    # "opt_cos": "opt_cosine_weights",
    # "opt_sin": "opt_sine_weights",
    # "opt_minus_sin": "opt_minus_sine_weights",

    # And finally extract the weights from the saved file and reformat them using the integration_weights_tools.
    # For this you just need to copy and paste the following lines at the beginning of the config, where the readout
    # parameters are defined as Python variables:
    # opt_weights = True
    # if opt_weights:
    #     from qualang_tools.config.integration_weights_tools import convert_integration_weights
    #
    #     weights = np.load("opt_weights.npz")
    #     opt_weights_real = convert_integration_weights(weights["weights_real"])
    #     opt_weights_minus_imag = convert_integration_weights(weights["weights_minus_imag"])
    #     opt_weights_imag = convert_integration_weights(weights["weights_imag"])
    #     opt_weights_minus_real = convert_integration_weights(weights["weights_minus_real"])
    # else:
    #     opt_weights_real = [(1.0, readout_len)]
    #     opt_weights_minus_imag = [(1.0, readout_len)]
    #     opt_weights_imag = [(1.0, readout_len)]
    #     opt_weights_minus_real = [(1.0, readout_len)]

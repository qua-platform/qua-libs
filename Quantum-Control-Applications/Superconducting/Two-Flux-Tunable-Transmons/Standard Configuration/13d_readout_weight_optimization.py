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
    - Update the integration weights in the state by following the steps at the end of the script.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.results import fetching_tool, progress_counter


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


###################
# The QUA program #
###################
n_avg = 1e4  # number of averages
# Set maximum readout duration for this scan and update the configuration accordingly
readout_len = readout_len
ringdown_len = 0 * u.us
# Set the sliced demod parameters
division_length = 10  # Size of each demodulation slice in clock cycles
number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)

with program() as opt_weights:
    n = declare(int)  # QUA variable for the averaging loop
    ind = declare(int)  # QUA variable for the index used to save each element in the 'I' & 'Q' vectors
    II = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'II'
    IQ = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'IQ'
    QI = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'QI'
    QQ = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the partial 'QQ'
    I = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the full 'I'=II+IQ
    Q = [declare(fixed, size=number_of_divisions) for _ in range(2)]  # QUA variable for the full 'Q'=QI+QQ
    II_st = [declare_stream() for _ in range(2)]  # Stream for the partial 'II'
    IQ_st = [declare_stream() for _ in range(2)]  # Stream for the partial 'IQ'
    QI_st = [declare_stream() for _ in range(2)]  # Stream for the partial 'QI'
    QQ_st = [declare_stream() for _ in range(2)]  # Stream for the partial 'QQ'
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Measure the ground state.
        wait(thermalization_time * u.ns)
        # Loop over the two resonators
        for rr, res in enumerate([1, 2]):
            # Save the sliced data (time trace of the demodulated data with a resolution equals to the division length)
            measure(
                "readout",
                f"rr{res}",
                None,
                demod.sliced("cos", II[rr], division_length, "out1"),
                demod.sliced("sin", IQ[rr], division_length, "out2"),
                demod.sliced("minus_sin", QI[rr], division_length, "out1"),
                demod.sliced("cos", QQ[rr], division_length, "out2"),
            )
            # Save the QUA vectors to their corresponding streams
            with for_(ind, 0, ind < number_of_divisions, ind + 1):
                save(II[rr][ind], II_st[rr])
                save(IQ[rr][ind], IQ_st[rr])
                save(QI[rr][ind], QI_st[rr])
                save(QQ[rr][ind], QQ_st[rr])

        # Measure the excited IQ blobs
        align()
        # Wait for the qubit to decay to the ground state
        wait(thermalization_time * u.ns)
        # Play the qubit drives
        play("x180", "q1_xy")
        play("x180", "q2_xy")
        align()
        # Loop over the two resonators
        for rr, res in enumerate([1, 2]):
            # Save the sliced data (time trace of the demodulated data with a resolution equals to the division length)
            measure(
                "readout",
                f"rr{res}",
                None,
                demod.sliced("cos", II[rr], division_length, "out1"),
                demod.sliced("sin", IQ[rr], division_length, "out2"),
                demod.sliced("minus_sin", QI[rr], division_length, "out1"),
                demod.sliced("cos", QQ[rr], division_length, "out2"),
            )
            # Save the QUA vectors to their corresponding streams
            with for_(ind, 0, ind < number_of_divisions, ind + 1):
                save(II[rr][ind], II_st[rr])
                save(IQ[rr][ind], IQ_st[rr])
                save(QI[rr][ind], QI_st[rr])
                save(QQ[rr][ind], QQ_st[rr])

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # Loop over the two resonators
        for q in range(2):
            II_st[q].buffer(2 * number_of_divisions).average().save(f"II_q{q}")
            IQ_st[q].buffer(2 * number_of_divisions).average().save(f"IQ_q{q}")
            QI_st[q].buffer(2 * number_of_divisions).average().save(f"QI_q{q}")
            QQ_st[q].buffer(2 * number_of_divisions).average().save(f"QQ_q{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

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
    ground_trace = [[], []]
    excited_trace = [[], []]
    norm_subtracted_trace = [[], []]
    weights_cos = [[], []]
    weights_sin = [[], []]
    weights_minus_sin = [[], []]
    weights_minus_cos = [[]]
    res_handles = job.result_handles
    for i in range(2):
        IIe, IIg = divide_array_in_half(res_handles.get(f"II_q{i}").fetch_all())
        IQe, IQg = divide_array_in_half(res_handles.get(f"IQ_q{i}").fetch_all())
        QIe, QIg = divide_array_in_half(res_handles.get(f"QI_q{i}").fetch_all())
        QQe, QQg = divide_array_in_half(res_handles.get(f"QQ_q{i}").fetch_all())
        # Sum the quadrature to fully demodulate the traces
        Ie = IIe + IQe
        Ig = IIg + IQg
        Qe = QIe + QQe
        Qg = QIg + QQg
        # Derive and normalize the ground and excited traces
        ground_trace[i] = Ig + 1j * Qg
        excited_trace[i] = Ie + 1j * Qe
        subtracted_trace = excited_trace[i] - ground_trace[i]
        norm_subtracted_trace[i] = normalize_complex_array(subtracted_trace)  # <- these are the optimal weights :)
        # Plot the results
        plot_three_complex_arrays(x_plot, ground_trace[i], excited_trace[i], norm_subtracted_trace[i])
        plt.suptitle(f"Integration weight optimization for qubit {i+1}")
        plt.tight_layout()

        weights_real = norm_subtracted_trace.real
        weights_minus_imag = -norm_subtracted_trace.imag
        weights_imag = norm_subtracted_trace.imag
        weights_minus_real = -norm_subtracted_trace.real
        # Save the weights for later use in the config
        np.savez(
            f"optimal_weights_q{i+1}",
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

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

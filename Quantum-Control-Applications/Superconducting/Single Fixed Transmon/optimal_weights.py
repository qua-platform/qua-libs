"""
optimal_weights.py: Optimal weights for the readout pulse
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.loops import from_array

###########
# Helpers #
###########


def divide_array_in_half(arr):
    split_index = len(arr) // 2
    arr1 = arr[:split_index]
    arr2 = arr[split_index:]
    return arr1, arr2


def create_complex_array(arr1, arr2):
    return arr1 + 1j * arr2


def subtract_complex_arrays(arr1, arr2):
    return arr1 - arr2


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
    ax1.plot(arr1.real, label="readl")
    ax1.plot(arr1.imag, label="imag")
    ax1.set_title("ground state")
    ax1.set_xlabel("Clock cicles")
    ax1.set_ylabel("demod traces [a.u.]")
    ax1.legend()
    ax2.plot(arr2.real, label="readl")
    ax2.plot(arr2.imag, label="imag")
    ax2.set_title("excited state")
    ax2.set_xlabel("Clock cicles")
    ax2.set_ylabel("demod traces [a.u.]")
    ax2.legend()
    ax3.plot(arr3.real, label="readl")
    ax3.plot(arr3.imag, label="imag")
    ax3.set_title("optimal weights")
    ax3.set_xlabel("Clock cicles")
    ax3.set_ylabel("substracted traces [a.u.]")
    ax3.legend()
    plt.show()


###################
# The QUA program #
###################
"""
To obtain the decay of the resonators the integration weights length needs to be set longer
than the readout_len, i.e., the pulse length. If integration weights and readout len
are the same then you will not get the ringdown of the resonator.
"""

division_length = 1  # in clock cycles
number_of_divisions = int(readout_len / (4 * division_length))
print("Integration weights chunk-size length in clock cyclces:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

n_avg = 1e4  # number of averages
cooldown_time = 5 * qubit_T1 // 4  # thermal decay time of the qubit

with program() as opt_weights:
    n = declare(int)
    II = declare(fixed, size=number_of_divisions)
    IQ = declare(fixed, size=number_of_divisions)
    QI = declare(fixed, size=number_of_divisions)
    QQ = declare(fixed, size=number_of_divisions)
    ind = declare(int)

    n_st = declare_stream()
    II_st = declare_stream()
    IQ_st = declare_stream()
    QI_st = declare_stream()
    QQ_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # ground state
        measure(
            "readout",
            "resonator",
            None,
            demod.sliced("cos", II, division_length, "out1"),
            demod.sliced("sin", IQ, division_length, "out2"),
            demod.sliced("minus_sin", QI, division_length, "out1"),
            demod.sliced("cos", QQ, division_length, "out2"),
        )

        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(II[ind], II_st)
            save(IQ[ind], IQ_st)
            save(QI[ind], QI_st)
            save(QQ[ind], QQ_st)

        wait(cooldown_time, "resonator")

        align()

        # excited state
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

        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(II[ind], II_st)
            save(IQ[ind], IQ_st)
            save(QI[ind], QI_st)
            save(QQ[ind], QQ_st)

        wait(cooldown_time, "resonator")

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
qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, opt_weights, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)

    job = qm.execute(opt_weights)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    res_handles = job.result_handles
    IIe, IIg = divide_array_in_half(res_handles.get("II").fetch_all())
    IQe, IQg = divide_array_in_half(res_handles.get("IQ").fetch_all())
    QIe, QIg = divide_array_in_half(res_handles.get("QI").fetch_all())
    QQe, QQg = divide_array_in_half(res_handles.get("QQ").fetch_all())

    Ie = IIe + IQe
    Ig = IIg + IQg
    Qe = QIe + QQe
    Qg = QIg + QQg

    ground_trace = create_complex_array(Ig, Qg)
    excited_trace = create_complex_array(Ie, Qe)
    substracted_trace = subtract_complex_arrays(excited_trace, ground_trace)
    norm_substracted_trace = normalize_complex_array(substracted_trace)  # <- these are the optimal weights :)
    plot_three_complex_arrays(ground_trace, excited_trace, norm_substracted_trace)

    # after obtaining the optimal weights, they need to be loaded to 'integration_weights' dictionary
    # in the config dictionary
    # for example
    # weights_plus_cos = norm_substracted_trace.real
    # weights_minus_sin = (-1) * norm_substracted_trace.imag
    # weights_sin = norm_substracted_trace.imag
    # weights_minus_cos = (-1) * norm_substracted_trace.real
    # then
    # config["integration_weights"]["opt_cos_weights"] = {"cosine": weights_plus_cos, "sine": weights_minus_sin}
    # config["integration_weights"]["opt_sin_weights"] = {"cosine": weights_sin, "sine": weights_cos}
    # config["integration_weights"]["opt_minus_sin_weights"] = {"cosine": weights_minus_sin, "sine": weights_minus_cos}
    # also need to add the new weights to readout_pulse
    # config['pulses']['readout_pulse']['integration_weights'] = ['opt_cos', 'opt_sin', 'opt_minus_sin']

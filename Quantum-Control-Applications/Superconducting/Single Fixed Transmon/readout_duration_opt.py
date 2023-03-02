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


def create_linear_array_of_same_length(arr):
    length = len(arr)
    return [i * 4 for i in range(length)]


def create_complex_array(arr1, arr2):
    return arr1 + 1j * arr2


def plot_complex_array(arr):
    plt.figure()
    plt.plot(arr.real)
    plt.plot(arr.imag)
    plt.show()


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
    ax1.plot(arr1.real)
    ax1.plot(arr1.imag)
    ax1.set_title("ground state")
    ax2.plot(arr2.real)
    ax2.plot(arr2.imag)
    ax2.set_title("excited state")
    ax3.plot(arr3.real)
    ax3.plot(arr3.imag)
    ax3.set_title("optimal weights")
    plt.show()


###################
# The QUA program #
###################

number_of_divisions = 250  # number of chunks to divide the integration weights into // put something meaningful here
division_length = int(readout_len / (4 * number_of_divisions))  # in clock cycles
print("Integration weights chunk-size length in clock cyclces:", division_length)

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
            demod.accumulated("cos", II, division_length, "out1"),
            demod.accumulated("sin", IQ, division_length, "out2"),
            demod.accumulated("minus_sin", QI, division_length, "out1"),
            demod.accumulated("cos", QQ, division_length, "out2"),
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
            demod.accumulated("cos", II, division_length, "out1"),
            demod.accumulated("sin", IQ, division_length, "out2"),
            demod.accumulated("minus_sin", QI, division_length, "out1"),
            demod.accumulated("cos", QQ, division_length, "out2"),
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
        # mean values
        II_st.buffer(2 * number_of_divisions).average().save("II")
        IQ_st.buffer(2 * number_of_divisions).average().save("IQ")
        QI_st.buffer(2 * number_of_divisions).average().save("QI")
        QQ_st.buffer(2 * number_of_divisions).average().save("QQ")
        # variances
        (
            ((II_st.buffer(2 * number_of_divisions) * II_st.buffer(2 * number_of_divisions)).average())
            - (II_st.buffer(2 * number_of_divisions).average() * II_st.buffer(2 * number_of_divisions).average())
        ).save("II_var")
        (
            ((IQ_st.buffer(2 * number_of_divisions) * IQ_st.buffer(2 * number_of_divisions)).average())
            - (IQ_st.buffer(2 * number_of_divisions).average() * IQ_st.buffer(2 * number_of_divisions).average())
        ).save("IQ_var")
        (
            ((QI_st.buffer(2 * number_of_divisions) * QI_st.buffer(2 * number_of_divisions)).average())
            - (QI_st.buffer(2 * number_of_divisions).average() * QI_st.buffer(2 * number_of_divisions).average())
        ).save("QI_var")
        (
            ((QQ_st.buffer(2 * number_of_divisions) * QQ_st.buffer(2 * number_of_divisions)).average())
            - (QQ_st.buffer(2 * number_of_divisions).average() * QQ_st.buffer(2 * number_of_divisions).average())
        ).save("QQ_var")

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
    IIe_var, IIg_var = divide_array_in_half(res_handles.get("II_var").fetch_all())
    IQe_var, IQg_var = divide_array_in_half(res_handles.get("IQ_var").fetch_all())
    QIe_var, QIg_var = divide_array_in_half(res_handles.get("QI_var").fetch_all())
    QQe_var, QQg_var = divide_array_in_half(res_handles.get("QQ_var").fetch_all())

    Ie = IIe + IQe
    Ig = IIg + IQg
    Qe = QIe + QQe
    Qg = QIg + QQg
    Ie_var = IIe_var + IQe_var
    Ig_var = IQg_var + IIg_var
    Qe_var = QIe_var + QQe_var
    Qg_var = QQg_var + QIg_var

    var = (Ie_var + Qe_var + Ig_var + Qg_var) / 4

    ground_trace = create_complex_array(Ig, Qg)
    excited_trace = create_complex_array(Ie, Qe)
    substracted_trace = subtract_complex_arrays(excited_trace, ground_trace)
    SNR = (np.abs(substracted_trace) ** 2) / (2 * var)
    plot_three_complex_arrays(ground_trace, excited_trace, SNR)

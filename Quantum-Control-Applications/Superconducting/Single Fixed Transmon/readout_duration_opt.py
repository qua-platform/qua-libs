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
    ax3.set_title("SNR")
    ax3.set_xlabel("Clock cicles")
    ax3.set_ylabel("substracted traces [a.u.]")
    ax3.legend()
    plt.show()


###################
# The QUA program #
###################
"""
If you want to obtain the behavior of the resonator during the ringdown, the ingtegration weights
length needs to be larger than the readout_pulse length.
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
    I = declare(fixed, size=number_of_divisions)
    Q = declare(fixed, size=number_of_divisions)
    ind = declare(int)

    n_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

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
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], I_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Q_st)

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
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], I_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Q_st)

        wait(cooldown_time, "resonator")

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # mean values
        I_st.buffer(2 * number_of_divisions).average().save("I")
        Q_st.buffer(2 * number_of_divisions).average().save("Q")
        # variances
        (
            ((I_st.buffer(2 * number_of_divisions) * I_st.buffer(2 * number_of_divisions)).average())
            - (I_st.buffer(2 * number_of_divisions).average() * I_st.buffer(2 * number_of_divisions).average())
        ).save("I_var")
        (
            ((Q_st.buffer(2 * number_of_divisions) * Q_st.buffer(2 * number_of_divisions)).average())
            - (Q_st.buffer(2 * number_of_divisions).average() * Q_st.buffer(2 * number_of_divisions).average())
        ).save("Q_var")

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

    Ie, Ig = divide_array_in_half(res_handles.get("I").fetch_all())
    Qe, Qg = divide_array_in_half(res_handles.get("Q").fetch_all())
    Ie_var, Ig_var = divide_array_in_half(res_handles.get("I_var").fetch_all())
    Qe_var, Qg_var = divide_array_in_half(res_handles.get("Q_var").fetch_all())

    var = (Ie_var + Qe_var + Ig_var + Qg_var) / 4

    ground_trace = create_complex_array(Ig, Qg)
    excited_trace = create_complex_array(Ie, Qe)
    SNR = (np.abs(excited_trace - ground_trace) ** 2) / (2 * var)
    plot_three_complex_arrays(ground_trace, excited_trace, SNR)

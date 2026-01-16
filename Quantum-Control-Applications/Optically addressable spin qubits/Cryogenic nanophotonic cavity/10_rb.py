"""
Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate fidelity
"""

from qm.qua import *
from qm import QuantumMachinesManager
from scipy.optimize import curve_fit
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qm import SimulationConfig
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
max_circuit_depth = 100
delta_depth = 1
num_of_sequences = 50
n_avg = 1
seed = 345324


###################################
# Helper functions and QUA macros #
###################################
def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(pi_len_NV // 4, "Yb")
            with case_(1):
                play("x180", "Yb")
            with case_(2):
                play("y180", "Yb")
            with case_(3):
                play("y180", "Yb")
                play("x180", "Yb")
            with case_(4):
                play("x90", "Yb")
                play("y90", "Yb")
            with case_(5):
                play("x90", "Yb")
                play("-y90", "Yb")
            with case_(6):
                play("-x90", "Yb")
                play("y90", "Yb")
            with case_(7):
                play("-x90", "Yb")
                play("-y90", "Yb")
            with case_(8):
                play("y90", "Yb")
                play("x90", "Yb")
            with case_(9):
                play("y90", "Yb")
                play("-x90", "Yb")
            with case_(10):
                play("-y90", "Yb")
                play("x90", "Yb")
            with case_(11):
                play("-y90", "Yb")
                play("-x90", "Yb")
            with case_(12):
                play("x90", "Yb")
            with case_(13):
                play("-x90", "Yb")
            with case_(14):
                play("y90", "Yb")
            with case_(15):
                play("-y90", "Yb")
            with case_(16):
                play("-x90", "Yb")
                play("y90", "Yb")
                play("x90", "Yb")
            with case_(17):
                play("-x90", "Yb")
                play("-y90", "Yb")
                play("x90", "Yb")
            with case_(18):
                play("x180", "Yb")
                play("y90", "Yb")
            with case_(19):
                play("x180", "Yb")
                play("-y90", "Yb")
            with case_(20):
                play("y180", "Yb")
                play("x90", "Yb")
            with case_(21):
                play("y180", "Yb")
                play("-x90", "Yb")
            with case_(22):
                play("x90", "Yb")
                play("y90", "Yb")
                play("x90", "Yb")
            with case_(23):
                play("-x90", "Yb")
                play("y90", "Yb")
                play("-x90", "Yb")


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "config": config,
}

###################
# The QUA program #
###################
with program() as rb:
    depth = declare(int)
    saved_gate = declare(int)
    m = declare(int)
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    state = declare(bool)
    state_st = declare_stream()
    times = declare(int, size=10)
    counts = declare(int)
    counts_st = declare_stream()

    with for_(m, 0, m < num_of_sequences, m + 1):
        sequence_list, inv_gate_list = generate_sequence()

        with for_(depth, 1, depth <= max_circuit_depth, depth + delta_depth):
            with for_(n, 0, n < n_avg, n + 1):
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                assign(saved_gate, sequence_list[depth])
                assign(sequence_list[depth], inv_gate_list[depth - 1])

                # initialization
                play("laser_ON", "F_transition")
                align()
                play("laser_ON", "A_transition")
                play("switch_ON", "excited_state_mw")
                align()

                # pulse sequence
                play_sequence(sequence_list, depth)

                align()  # global align

                # readout laser
                play("laser_ON", "A_transition", duration=int(meas_len // 4))
                # does it needs buffer to prevent damage?

                align()

                # decay readout
                measure("readout", "SNSPD", None, time_tagging.analog(times, meas_len, counts))
                save(counts, counts_st)  # save counts
                wait(100)

                assign(sequence_list[depth], saved_gate)

    with stream_processing():
        counts_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_of_sequences, max_circuit_depth).save("res")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, rb, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))

else:
    qm = qmm.open_qm(config)
    job = qm.execute(rb)
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    state = res_handles.res.fetch_all()

    # value = 1 - np.average(state, axis=0)
    value = np.average(state, axis=0)

    def power_law(m, a, b, p):
        return a * (p**m) + b

    fig = plt.figure()
    x = np.linspace(1, max_circuit_depth, max_circuit_depth)
    plt.plot(x, value)
    plt.xlabel("Number of cliffords")
    plt.ylabel("Sequence Fidelity")

    pars, cov = curve_fit(
        f=power_law,
        xdata=x,
        ydata=value,
        p0=[0.5, 0.5, 0.9],
        bounds=(-np.inf, np.inf),
        maxfev=2000,
    )

    plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)

    stdevs = np.sqrt(np.diag(cov))

    print("#########################")
    print("### Fitted Parameters ###")
    print("#########################")
    print(f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})")
    print("Covariance Matrix")
    print(cov)

    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / 1.875

    print("#########################")
    print("### Useful Parameters ###")
    print("#########################")
    print(
        f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1})\n"
        f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1})\n"
        f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
    )

    np.savez("rb_values", value)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"fig_live": fig})
    save_data_dict.update({"value_data": value})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

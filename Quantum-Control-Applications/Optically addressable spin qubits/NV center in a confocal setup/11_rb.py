from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from scipy.optimize import curve_fit
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

# Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate fidelity

##############################
# Program-specific variables #
##############################
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
max_circuit_depth = 1000
num_of_sequences = 500
n_avg = 20
seed = 345324
delta_clifford = 10  # Must be > 1
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."

def power_law(power, a, b, p):
    return a * (p**power) + b

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
                wait(pi_len_NV * u.ns, "NV")
            with case_(1):
                play("x180", "NV")
            with case_(2):
                play("y180", "NV")
            with case_(3):
                play("y180", "NV")
                play("x180", "NV")
            with case_(4):
                play("x90", "NV")
                play("y90", "NV")
            with case_(5):
                play("x90", "NV")
                play("-y90", "NV")
            with case_(6):
                play("-x90", "NV")
                play("y90", "NV")
            with case_(7):
                play("-x90", "NV")
                play("-y90", "NV")
            with case_(8):
                play("y90", "NV")
                play("x90", "NV")
            with case_(9):
                play("y90", "NV")
                play("-x90", "NV")
            with case_(10):
                play("-y90", "NV")
                play("x90", "NV")
            with case_(11):
                play("-y90", "NV")
                play("-x90", "NV")
            with case_(12):
                play("x90", "NV")
            with case_(13):
                play("-x90", "NV")
            with case_(14):
                play("y90", "NV")
            with case_(15):
                play("-y90", "NV")
            with case_(16):
                play("-x90", "NV")
                play("y90", "NV")
                play("x90", "NV")
            with case_(17):
                play("-x90", "NV")
                play("-y90", "NV")
                play("x90", "NV")
            with case_(18):
                play("x180", "NV")
                play("y90", "NV")
            with case_(19):
                play("x180", "NV")
                play("-y90", "NV")
            with case_(20):
                play("y180", "NV")
                play("x90", "NV")
            with case_(21):
                play("y180", "NV")
                play("-x90", "NV")
            with case_(22):
                play("x90", "NV")
                play("y90", "NV")
                play("x90", "NV")
            with case_(23):
                play("-x90", "NV")
                play("y90", "NV")
                play("-x90", "NV")


###################
# The QUA program #
###################
with program() as rb:
    depth = declare(int)
    depth_target = declare(int)
    saved_gate = declare(int)
    m = declare(int)
    n = declare(int)
    counts = declare(int)
    times = declare(int, size=100)
    counts_st = declare_stream()
    times_st = declare_stream()
    m_st = declare_stream()

    with for_(m, 0, m < num_of_sequences, m + 1):
        sequence_list, inv_gate_list = generate_sequence()

        assign(depth_target, 0)

        with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])

            with if_((depth == 1) | (depth == depth_target)):
                
                play("laser_ON", "AOM1")
                wait(100 * u.ns, "AOM1")

                with for_(n, 0, n < n_avg, n + 1):
                    play_sequence(sequence_list, depth)
                    align()
                    play("laser_ON", "AOM1")
                    measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
                    save(counts, counts_st)  # save counts
                    wait(100 * u.ns)  # wait in between iterations

                assign(depth_target, depth_target + delta_clifford)

            assign(sequence_list[depth], saved_gate)
        save(m, m_st)

    with stream_processing():
        m_st.save("iteration")
        counts_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).average().save('counts')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=20_000)  # in clock cycles
    job = qmm.simulate(config, rb, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)

    job = qm.execute(rb)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    # data analysis
    x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
    x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment
    while results.is_processing():
        # data analysis
        counts, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, num_of_sequences, start_time=results.get_start_time())
        value_avg = counts
        # need renormalization to contrast to be able to do fitting
        error_avg = np.std(counts)
        # Plot results
        plt.cla()
        plt.errorbar(x, value_avg, yerr=error_avg, marker=".")
        # pars, cov = curve_fit(
        #     f=power_law,
        #     xdata=x,
        #     ydata=value_avg,
        #     p0=[0.5, 0.5, 0.9],
        #     bounds=(-np.inf, np.inf),
        #     maxfev=2000,
        # )
        # plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)
        plt.xlabel("Number of Clifford gates")
        plt.ylabel("Sequence Fidelity")
        plt.title("Single qubit RB")
        plt.pause(0.1)

    # stdevs = np.sqrt(np.diag(cov))

    # print("#########################")
    # print("### Fitted Parameters ###")
    # print("#########################")
    # print(f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})")
    # print("Covariance Matrix")
    # print(cov)

    # one_minus_p = 1 - pars[2]
    # r_c = one_minus_p * (1 - 1 / 2**1)
    # r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
    # r_c_std = stdevs[2] * (1 - 1 / 2**1)
    # r_g_std = r_c_std / 1.875

    # print("#########################")
    # print("### Useful Parameters ###")
    # print("#########################")
    # print(
    #     f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1})\n"
    #     f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1})\n"
    #     f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
    # )

    # np.savez("rb_values", value)
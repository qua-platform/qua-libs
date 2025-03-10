"""
        SINGLE QUBIT RANDOMIZED BENCHMARKING (for gates >= 40ns)
The program consists in playing random sequences of Clifford gates and measuring the state of the resonator afterwards.
Each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.

Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate
.
Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Having set the pi pulse amplitude and duration in the configuration
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from scipy.optimize import curve_fit
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
num_of_sequences = 50  # Number of random sequences
n_avg = 20  # Number of averaging loops for each random sequence
max_circuit_depth = 1000  # Maximum circuit depth
delta_clifford = 10  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
seed = 345324  # Pseudo-random number generator seed
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


###################################
# Helper functions and QUA macros #
###################################
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
                wait(x180_len_NV * u.ns, "NV")
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
# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "config": config,
}

###################
# The QUA program #
###################
with program() as rb:
    depth = declare(int)  # QUA variable for the varying depth
    depth_target = declare(int)  # QUA variable for the the current depth (changes in steps of delta_clifford)
    # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    saved_gate = declare(int)
    m = declare(int)  # QUA variable for the loop over random sequences
    n = declare(int)  # QUA variable for the averaging loop
    counts = declare(int)  # saves number of photon counts
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    # The relevant streams
    m_st = declare_stream()
    counts_st = declare_stream()
    times_st = declare_stream()

    with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
        sequence_list, inv_gate_list = generate_sequence()  # Generate the random sequence of length max_circuit_depth

        assign(depth_target, 0)  # Initialize the current depth to 0
        with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])
            # Only played the depth corresponding to target_depth
            with if_((depth == 1) | (depth == depth_target)):
                # Spin initialization
                play("laser_ON", "AOM1")
                wait(wait_for_initialization * u.ns, "AOM1")

                with for_(n, 0, n < n_avg, n + 1):
                    # The strict_timing ensures that the sequence will be played without gaps
                    with strict_timing_():
                        play_sequence(sequence_list, depth)
                    align()  # Play the laser pulse after the Echo sequence
                    # Measure and detect the photons on SPCM1
                    play("laser_ON", "AOM1")
                    measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
                    save(counts, counts_st)  # save counts
                    wait(wait_between_runs * u.ns)  # wait in between iterations

                # Go to the next depth
                assign(depth_target, depth_target + delta_clifford)
            # Reset the last gate of the sequence back to the original Clifford gate
            # (that was replaced by the recovery gate at the beginning)
            assign(sequence_list[depth], saved_gate)
        # Save the counter for the progress bar
        save(m, m_st)

    with stream_processing():
        m_st.save("iteration")
        # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
        counts_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).buffer(
            num_of_sequences
        ).save("counts")
        # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
        counts_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).average().save(
            "counts_avg"
        )

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

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
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rb)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts_avg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    # data analysis
    x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
    x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment
    while results.is_processing():
        # data analysis
        counts_avg, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, num_of_sequences, start_time=results.get_start_time())
        # Plot averaged values
        plt.cla()
        plt.plot(x, counts_avg, marker=".")
        plt.xlabel("Number of Clifford gates")
        plt.ylabel("Sequence Fidelity")
        plt.title("Single qubit RB")
        plt.pause(0.1)
    # At the end of the program, fetch the non-averaged results to get the error-bars

    results = fetching_tool(job, data_list=["counts"])
    counts = results.fetch_all()[0]
    value_avg = np.mean(counts, axis=0)
    error_avg = np.std(counts, axis=0)
    # data analysis
    pars, cov = curve_fit(
        f=power_law,
        xdata=x,
        ydata=value_avg,
        p0=[0.5, 0.5, 0.9],
        bounds=(-np.inf, np.inf),
        maxfev=2000,
    )
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

    # Plots
    plt.figure()
    plt.errorbar(x, value_avg, yerr=error_avg, marker=".")
    plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)
    plt.xlabel("Number of Clifford gates")
    plt.ylabel("Sequence Fidelity")
    plt.title("Single qubit RB")

    # np.savez("rb_values", value)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts_data": counts})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

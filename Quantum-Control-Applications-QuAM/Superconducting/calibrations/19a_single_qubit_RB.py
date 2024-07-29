# %%
"""
SINGLE QUBIT RANDOMIZED BENCHMARKING (for gates >= 40ns)
The program consists in playing random sequences of Clifford gates and measuring the state of the resonator afterwards.
Each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.

If the readout has been calibrated and is good enough, then state discrimination can be applied to only return the state
of the qubit. Otherwise, the 'I' and 'Q' quadratures are returned.
Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate
.
Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.
"""

from pathlib import Path

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.units import unit
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import node_save

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

import matplotlib

matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits

##############################
# Program-specific variables #
##############################
num_of_sequences = 50  # Number of random sequences
n_avg = 10  # Number of averaging loops for each random sequence
max_circuit_depth = 1000  # Maximum circuit depth
delta_clifford = 10  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = 345324  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = False
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


def play_sequence(sequence_list, depth, qubit: Transmon):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
            with case_(1):
                qubit.xy.play("x180")
            with case_(2):
                qubit.xy.play("y180")
            with case_(3):
                qubit.xy.play("y180")
                qubit.xy.play("x180")
            with case_(4):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
            with case_(5):
                qubit.xy.play("x90")
                qubit.xy.play("-y90")
            with case_(6):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
            with case_(7):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
            with case_(8):
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(9):
                qubit.xy.play("y90")
                qubit.xy.play("-x90")
            with case_(10):
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(11):
                qubit.xy.play("-y90")
                qubit.xy.play("-x90")
            with case_(12):
                qubit.xy.play("x90")
            with case_(13):
                qubit.xy.play("-x90")
            with case_(14):
                qubit.xy.play("y90")
            with case_(15):
                qubit.xy.play("-y90")
            with case_(16):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(17):
                qubit.xy.play("-x90")
                qubit.xy.play("-x90")
                qubit.xy.play("x90")
            with case_(18):
                qubit.xy.play("x180")
                qubit.xy.play("y90")
            with case_(19):
                qubit.xy.play("x180")
                qubit.xy.play("-y90")
            with case_(20):
                qubit.xy.play("y180")
                qubit.xy.play("x90")
            with case_(21):
                qubit.xy.play("y180")
                qubit.xy.play("-x90")
            with case_(22):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(23):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("-x90")


###################
# The QUA program #
###################
def get_rb_program(qubit: Transmon):
    with program() as rb:
        depth = declare(int)  # QUA variable for the varying depth
        depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        n = declare(int)  # QUA variable for the averaging loop
        I = declare(fixed)  # QUA variable for the 'I' quadrature
        Q = declare(fixed)  # QUA variable for the 'Q' quadrature
        state = declare(bool)  # QUA variable for state discrimination
        # The relevant streams
        m_st = declare_stream()
        I_st = declare_stream()
        Q_st = declare_stream()
        if state_discrimination:
            state_st = declare_stream()

        # Bring the active qubits to the minimum frequency point
        machine.apply_all_flux_to_min()

        with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
            (
                sequence_list,
                inv_gate_list,
            ) = generate_sequence()  # Generate the random sequence of length max_circuit_depth

            assign(depth_target, 0)  # Initialize the current depth to 0

            with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                assign(saved_gate, sequence_list[depth])
                assign(sequence_list[depth], inv_gate_list[depth - 1])
                # Only played the depth corresponding to target_depth
                with if_((depth == 1) | (depth == depth_target)):
                    with for_(n, 0, n < n_avg, n + 1):
                        # Can replace by active reset
                        qubit.resonator.wait(machine.thermalization_time * u.ns)
                        # Align the two elements to play the sequence after qubit initialization
                        qubit.resonator.align(qubit.xy.name)
                        # The strict_timing ensures that the sequence will be played without gaps
                        with strict_timing_():
                            # Play the random sequence of desired depth
                            play_sequence(sequence_list, depth, qubit)
                        # Align the two elements to measure after playing the circuit.
                        qubit.resonator.align(qubit.xy.name)
                        # # Play through the 2nd resonator to be in the same condition as when the readout was optimized
                        # if qubit.resonator == q1.resonator:
                        #     q2.resonator.play("readout")
                        # else:
                        #     q1.resonator.play("readout")
                        # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                        qubit.resonator.measure("readout", qua_vars=(I, Q))
                        save(I, I_st)
                        save(Q, Q_st)
                        # Make sure you updated the ge_threshold
                        if state_discrimination:
                            assign(
                                state,
                                I > qubit.resonator.operations["readout"].threshold,
                            )
                            save(state, state_st)

                    # Go to the next depth
                    assign(depth_target, depth_target + delta_clifford)
                # Reset the last gate of the sequence back to the original Clifford gate
                # (that was replaced by the recovery gate at the beginning)
                assign(sequence_list[depth], saved_gate)
            # Save the counter for the progress bar
            save(m, m_st)

        with stream_processing():
            m_st.save("iteration")
            if state_discrimination:
                # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
                state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(
                    num_of_sequences
                ).save("state")
                # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
                state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).average().save(
                    "state_avg"
                )
            else:
                I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save("I")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save("Q")
                I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).average().save("I_avg")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).average().save("Q_avg")

    return rb


simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, get_rb_program(qubits[0]), simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Prepare data for saving
    data = {}
    for qubit in qubits:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Calibrate the active qubits
        # machine.calibrate_octave_ports(qm)
        # Send the QUA program to the OPX, which compiles and executes it
        rb_program = get_rb_program(qubit)
        job = qm.execute(rb_program)
        # Get results from QUA program
        if state_discrimination:
            results = fetching_tool(job, data_list=["state_avg", "iteration"], mode="live")
        else:
            results = fetching_tool(job, data_list=["I_avg", "Q_avg", "iteration"], mode="live")
        # Live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        # data analysis
        x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
        x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment
        while results.is_processing():
            # data analysis
            if state_discrimination:
                state_avg, iteration = results.fetch_all()
                value_avg = state_avg
            else:
                I, Q, iteration = results.fetch_all()
                value_avg = I

            # Progress bar
            progress_counter(iteration, num_of_sequences, start_time=results.get_start_time())
            # Plot averaged values
            plt.cla()
            plt.plot(x, value_avg, marker=".")
            plt.xlabel("Number of Clifford gates")
            plt.ylabel("Sequence Fidelity")
            plt.title("Single qubit RB")
            plt.pause(0.1)

        # At the end of the program, fetch the non-averaged results to get the error-bars
        if state_discrimination:
            results = fetching_tool(job, data_list=["state"])
            state = results.fetch_all()[0]
            value_avg = np.mean(state, axis=0)
            error_avg = np.std(state, axis=0)
        else:
            results = fetching_tool(job, data_list=["I", "Q"])
            I, Q = results.fetch_all()
            value_avg = np.mean(I, axis=0)
            error_avg = np.std(I, axis=0)
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
        print(
            f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})"
        )
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
        fig_analysis = plt.figure()
        plt.errorbar(x, value_avg, yerr=error_avg, marker=".")
        plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)
        plt.xlabel("Number of Clifford gates")
        plt.ylabel("Sequence Fidelity")
        plt.title(f"Single qubit RB for {qubit.name}")

        plt.show()
        # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
        qm.close()

        # Save data from the node
        data[f"{qubit.name}_depth"] = x
        data[f"{qubit.name}_fidelity"] = value_avg
        data[f"{qubit.name}_error"] = error_avg
        data[f"{qubit.name}_fit"] = power_law(x, *pars)
        data[f"{qubit.name}_covariance_par"] = pars
        data[f"{qubit.name}_error_rate"] = one_minus_p
        data[f"{qubit.name}_clifford_set_infidelity"] = r_c
        data[f"{qubit.name}_gate_infidelity"] = r_g
        data[f"{qubit.name}_figure"] = fig
        data[f"{qubit.name}_figure_analysis"] = fig_analysis

    node_save(machine, "randomized_benchmarking", data, additional_files=True)

# %%

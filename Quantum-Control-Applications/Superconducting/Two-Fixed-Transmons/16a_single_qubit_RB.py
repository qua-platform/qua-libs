# %%
"""
        SINGLE QUBIT RANDOMIZED BENCHMARKING
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration_mw_fem import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import multiplexed_readout, qua_declaration, active_reset
import math
from qualang_tools.results.data_handler import DataHandler
from scipy.optimize import curve_fit
import matplotlib
import time

matplotlib.use('TkAgg')

##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
num_of_sequences = 50  # Number of random sequences
n_avg = 10  # Number of averaging loops for each random sequence
max_circuit_depth = 2000  # Maximum circuit depth
delta_clifford = 20  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
seed = 345324  # Pseudo-random number generator seed
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
# qubits = [f"q{i}_xy" for i in [qc, qt]]
# resonators = [f"q{i}_rr" for i in [qc, qt]]
qubits = [qb for qb in QUBIT_CONSTANTS.keys()]
qubits_to_play = ['q4_xy']
resonators = [key for key in RR_CONSTANTS.keys()]

# Assertion
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "config": config,
    "num_of_sequences": num_of_sequences,
    "delta_clifford": delta_clifford,
    "max_circuit_depth": max_circuit_depth,
    "weights": weights,
    "reset_method": reset_method,
}


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


def play_sequence(sequence_list, depth, qb):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(PI_LEN // 4, qb)
            with case_(1):
                play("x180", qb)
            with case_(2):
                play("y180", qb)
            with case_(3):
                play("y180", qb)
                play("x180", qb)
            with case_(4):
                play("x90", qb)
                play("y90", qb)
            with case_(5):
                play("x90", qb)
                play("-y90", qb)
            with case_(6):
                play("-x90", qb)
                play("y90", qb)
            with case_(7):
                play("-x90", qb)
                play("-y90", qb)
            with case_(8):
                play("y90", qb)
                play("x90", qb)
            with case_(9):
                play("y90", qb)
                play("-x90", qb)
            with case_(10):
                play("-y90", qb)
                play("x90", qb)
            with case_(11):
                play("-y90", qb)
                play("-x90", qb)
            with case_(12):
                play("x90", qb)
            with case_(13):
                play("-x90", qb)
            with case_(14):
                play("y90", qb)
            with case_(15):
                play("-y90", qb)
            with case_(16):
                play("-x90", qb)
                play("y90", qb)
                play("x90", qb)
            with case_(17):
                play("-x90", qb)
                play("-y90", qb)
                play("x90", qb)
            with case_(18):
                play("x180", qb)
                play("y90", qb)
            with case_(19):
                play("x180", qb)
                play("-y90", qb)
            with case_(20):
                play("y180", qb)
                play("x90", qb)
            with case_(21):
                play("y180", qb)
                play("-x90", qb)
            with case_(22):
                play("x90", qb)
                play("y90", qb)
                play("x90", qb)
            with case_(23):
                play("-x90", qb)
                play("y90", qb)
                play("-x90", qb)



###################
#   QUA Program   #
###################

with program() as PROGRAM:
    depth = declare(int)  # QUA variable for the varying depth
    depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
    # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    saved_gate = declare(int)
    m = declare(int)  # QUA variable for the loop over random sequences
    m_st = declare_stream()
    I, I_st, Q, Q_st, n, _ = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    state_st = [declare_stream() for _ in range(len(resonators))]

    with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
        # Save the counter for the progress bar
        save(m, m_st)
        sequence_list, inv_gate_list = generate_sequence()  # Generate the random sequence of length max_circuit_depth

        assign(depth_target, 0)  # Initialize the current depth to 0

        with for_(depth, 1, depth <= max_circuit_depth, depth + 1):  # Loop over the depths
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])
            # Only played the depth corresponding to target_depth

            with if_((depth == 1) | (depth == depth_target)):

                with for_(n, 0, n < n_avg, n + 1):  # Averaging loop
                    if reset_method == "wait":
                        wait(qb_reset_time >> 2)
                    elif reset_method == "active":
                        global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights,delay=187)
                    # Align the two elements to play the sequence after qubit initialization
                    align()
                    # The strict_timing ensures that the sequence will be played without gaps
                    for qb in qubits_to_play:
                        # with strict_timing_():
                        # Play the random sequence of desired depth
                        play_sequence(sequence_list, depth, qb)
                    # Align the two elements to measure after playing the circuit.
                    align()
                    multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators=resonators, weights=weights)

                # Go to the next depth
                assign(depth_target, depth_target + delta_clifford)

            # Reset the last gate of the sequence back to the original Clifford gate
            # (that was replaced by the recovery gate at the beginning)
            assign(sequence_list[depth], saved_gate)

    with stream_processing():
        m_st.save("iteration")
        # NOTE: need to discuss if we do qubit by qubit or multiplexed
        for ind, rr in enumerate(resonators):
            # I_st[ind].buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).buffer(
            #     num_of_sequences
            # ).save(f"I_{rr}")
            # Q_st[ind].buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).buffer(
            #     num_of_sequences
            # ).save(f"Q_{rr}")
            # state_st[ind].boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
            #     max_circuit_depth / delta_clifford + 1
            # ).buffer(num_of_sequences).save(f"state_{rr}")
            
            I_st[ind].buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).average().save(
                f"I_avg_{rr}"
            )
            Q_st[ind].buffer(n_avg).map(FUNCTIONS.average()).buffer(max_circuit_depth / delta_clifford + 1).average().save(
                f"Q_avg_{rr}"
            )
            state_st[ind].boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
                max_circuit_depth / delta_clifford + 1
            ).average().save(f"state_avg_{rr}")


if __name__ == "__main__":
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
        job = qmm.simulate(config, PROGRAM, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.plot(block=False)
    else:
        try:
            # Open the quantum machine
            qm = qmm.open_qm(config)
            
            # Send the QUA program to the OPX, which compiles and executes it
            job = qm.execute(PROGRAM, flags=['not-strict-timing'])

            fetch_names = ["iteration"]
            for rr in resonators:
                # fetch_names.append(f"I_{rr}")
                # fetch_names.append(f"Q_{rr}")
                # fetch_names.append(f"state_{rr}")
                fetch_names.append(f"I_avg_{rr}")
                fetch_names.append(f"Q_avg_{rr}")
                fetch_names.append(f"state_avg_{rr}")
            # Tool to easily fetch results from the OPX (results_handle used in it)
            results = fetching_tool(job, fetch_names, mode="live")
            # Prepare the figure for live plotting
            fig = plt.figure()
            interrupt_on_close(fig, job)
            # Data analysis and plotting
            num_resonators = len(resonators)
            num_rows = math.ceil(math.sqrt(num_resonators))
            num_cols = math.ceil(num_resonators / num_rows)
            # Live plotting

            x = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
            x[0] = 1  # to set the first value of 'x' to be depth = 1 as in the experiment

            while results.is_processing():

                # Fetch results
                res = results.fetch_all()
                # Progress bar
                progress_counter(res[0], num_of_sequences, start_time=results.start_time)

                plt.suptitle("RB")
                for ind, (qb, rr) in enumerate(zip(qubits, resonators)):

                    S = res[3*ind + 3]
                    # Plot
                    plt.subplot(num_rows, num_cols, ind + 1)
                    plt.cla()
                    plt.plot(x, S, marker=".", color='r')
                    plt.xlabel("Number of Clifford gates")
                    plt.ylabel("Sequence Fidelity")
                    plt.title(f"Qb - {qb}")

                plt.tight_layout()
                plt.pause(2)

            # data analysis
            ind = 3
            value_avg = res[3*ind + 3]
            pars, cov = curve_fit(
                f=power_law,
                xdata=x,
                ydata=value_avg,
                p0=[0.5, 0.5, 0.9],
                bounds=(-np.inf, np.inf),
                maxfev=2000,
            )
            stdevs = np.sqrt(np.diag(cov))

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
                f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})\n"
                f"Gate fidelity: = {np.format_float_scientific((1-r_g)*100, precision=3)}  ({r_g_std:.1})"
            )

            for ind, rr in enumerate(resonators):
                # save_data_dict[rr+"_I"] = res[6*ind + 1]
                # save_data_dict[rr+"_Q"] = res[6*ind + 2]
                # save_data_dict[rr+"_state"] = res[6*ind + 3]
                save_data_dict[rr+"_I_avg"] = res[3*ind + 1]
                save_data_dict[rr+"_Q_avg"] = res[3*ind + 2]
                save_data_dict[rr+"_state_avg"] = res[3*ind + 3]

            elapsed_time = time.time() - results.start_time
            save_data_dict["elapsed_time"] = elapsed_time
            save_data_dict["error_rate"] = one_minus_p
            save_data_dict['clifford_infidelity'] = r_c
            save_data_dict['gate_infidelity'] = r_g
            save_data_dict['gate_fidelity'] = (1-r_g)*100
            save_data_dict['index_to_fit'] = ind

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="rb")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
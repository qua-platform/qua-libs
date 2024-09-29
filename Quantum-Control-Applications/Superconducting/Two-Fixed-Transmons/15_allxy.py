# %%
"""
        ALL-XY MEASUREMENT
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout, active_reset
from qualang_tools.results.data_handler import DataHandler
import math
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
n_avg = 10_000

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
# assert n_avg <= 10_000, "revise your number of shots"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "config": config,
}


# All XY sequences. The sequence names must match corresponding operation in the config
sequence = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]


def allXY(pulses, qb):
    """
    Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
    **Example:** I, Q = allXY(['I', 'y90'])

    :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
        operations in the config except for the identity operation that must be called 'I'.
    """
    if pulses[0] != "I":
        play(pulses[0], qb)  # Either play the sequence
    else:
        wait(PI_LEN >> 2, qb)  # or wait if sequence is identity
    # Play the 2nd gate or wait if the gate is identity
    if pulses[1] != "I":
        play(pulses[1], qb)  # Either play the sequence
    else:
        wait(PI_LEN >> 2, qb)  # or wait if sequence is identity


##################
#   Parameters   #
##################
###################
#   QUA Program   #
###################

with program() as PROGRAM:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    state_st = [declare_stream() for _ in range(len(resonators))]

    with for_(n, 0, n < n_avg, n + 1):

        save(n, n_st)
        for i in range(len(sequence)):

            align()
            for qb in qubits_to_play:
                allXY(sequence[i], qb)

            align()
            multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators=resonators, weights=weights)
            # Wait for the qubit to decay to the ground state - Can be replaced by active reset
            if reset_method == "wait":
                wait(qb_reset_time >> 2)
            elif reset_method == "active":
                global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)

    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(21).average().save(f"I_{rr}")
            Q_st[ind].buffer(21).average().save(f"Q_{rr}")
            state_st[ind].boolean_to_int().buffer(21).average().save(f"state_{rr}")


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
        plt.show(block=False)
    else:
        try:
            # Open the quantum machine
            qm = qmm.open_qm(config)
            
            # Send the QUA program to the OPX, which compiles and executes it
            job = qm.execute(PROGRAM)
            fetch_names = ["iteration"]
            for rr in resonators:
                fetch_names.append(f"I_{rr}")
                fetch_names.append(f"Q_{rr}")
                fetch_names.append(f"state_{rr}")
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
            while results.is_processing():
                # Fetch results
                res = results.fetch_all()
                # Progress bar
                progress_counter(res[0], n_avg, start_time=results.start_time)

                plt.suptitle("AllXY")
                for ind, (qb, rr) in enumerate(zip(qubits, resonators)):

                    S = res[3*ind+3]  # res[3*ind+1] + 1j * res[3*ind+2]
                    save_data_dict[f"I_{rr}"] = res[3*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[3*ind + 2]
                    save_data_dict[rr+"_state"] = res[3*ind + 3]

                    # Plot
                    plt.subplot(num_rows, num_cols, ind + 1)
                    plt.cla()
                    plt.plot(-2 * S + 1 , "bx", label="Experimental data")
                    plt.plot([1.0] * 5 + [0.0] * 12 + [-1.0] * 4, "r-", label="Expected value")
                    plt.ylabel("State")
                    plt.xticks(ticks=range(len(sequence)), labels=[f"{_}" for _ in sequence], rotation=45)
                    plt.title(f"Qb - {qb}")

                plt.tight_layout()
                plt.pause(2)

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="allxy")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
"""
        ALL-XY MEASUREMENT
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 10
qubit = "q1_xy"

# Data to save
save_data_dict = {"n_avg": n_avg, "config": config, "qubit": qubit}

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
        wait(pi_len // 4, qb)  # or wait if sequence is identity
    # Play the 2nd gate or wait if the gate is identity
    if pulses[1] != "I":
        play(pulses[1], qb)  # Either play the sequence
    else:
        wait(pi_len // 4, qb)  # or wait if sequence is identity


###################
#   QUA Program   #
###################

with program() as PROGRAM:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)

    with for_(n, 0, n < n_avg, n + 1):

        save(n, n_st)

        for i in range(len(sequence)):

            align()

            allXY(sequence[i], qubit)

            align()

            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
            # Wait for the qubit to decay to the ground state - Can be replaced by active reset
            wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("iteration")
        for ind in range(2):
            I_st[ind].buffer(21).average().save(f"I{ind+1}")
            Q_st[ind].buffer(21).average().save(f"Q{ind+1}")

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
        # Prepare the figure for live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, ["iteration", "I1", "Q1", "I2", "Q2"], mode="live")
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], n_avg, start_time=results.start_time)

            plt.suptitle("AllXY")

            if qubit == "q1_xy":
                ind = 0
            else:
                ind = 1

            I = res[2 * ind + 1]
            Q = res[2 * ind + 2]

            save_data_dict[f"I{ind}"] = res[2 * ind + 1]
            save_data_dict[f"Q{ind}"] = res[2 * ind + 2]

            # Plot
            plt.suptitle(f"All XY for qubit {ind}")
            plt.subplot(211)
            plt.cla()
            plt.plot(I, "bx", label="Experimental data")
            plt.plot([np.max(I)] * 5 + [(np.mean(I))] * 12 + [np.min(I)] * 4, "r-", label="Expected value")
            plt.ylabel("I quadrature [a.u.]")
            plt.xticks(ticks=range(len(sequence)), labels=["" for _ in sequence], rotation=45)
            plt.legend()
            plt.subplot(212)
            plt.cla()
            plt.plot(Q, "bx", label="Experimental data")
            plt.plot([np.max(Q)] * 5 + [(np.mean(Q))] * 12 + [np.min(Q)] * 4, "r-", label="Expected value")
            plt.ylabel("Q quadrature [a.u.]")
            plt.xticks(ticks=range(len(sequence)), labels=[str(el) for el in sequence], rotation=45)
            plt.legend()

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

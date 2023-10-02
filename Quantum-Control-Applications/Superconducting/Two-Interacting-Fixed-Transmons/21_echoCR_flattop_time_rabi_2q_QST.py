"""
Reference: J. M. Chow et al., Phys. Rev. Letters 107, 080502 (2011)
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout


def two_qb_QST(qb1: str, qb2: str, len1: float, len2: float):
    """
    QUA macro to do two-qubit quantum state tomography
    """
    with switch_(c):
        with case_(0):
            play("-y90", qb1)
            play("-y90", qb2)
        with case_(1):
            play("y-90", qb1)
            play("-x90", qb2)
        with case_(2):
            play("-x90", qb1)
            play("-y90", qb2)
        with case_(3):
            play("-x90", qb1)
            play("-x90", qb2)
        with case_(4):
            play("-y90", qb1)
            wait(int(len2 * 1e9 // 4), qb2)
        with case_(5):
            wait(int(len1 * 1e9 // 4), qb1)
            play("-y90", qb2)
        with case_(6):
            play("-x90", qb1)
            wait(int(len2 * 1e9 // 4), qb2)
        with case_(7):
            wait(int(len1 * 1e9 // 4), qb1)
            play("-x90", qb2)
        with case_(8):
            wait(int(len1 * 1e9 // 4), qb1)
            wait(int(len2 * 1e9 // 4), qb2)


def plot_tomography_results(data_q1, data_q2, xaxis, fig=None, axs=None):
    """
    Helper function to display quantum state tomography data
    """
    # Define the column titles
    col_titles = [
        "<-Y/2-Y/2>",
        "<-Y/2-X/2>",
        "<-X/2-Y/2>",
        "<-X/2-X/2>",
        "<-Y/2,I>",
        "<I,-Y/2>",
        "<-X/2,I>",
        "<I,-X/2>",
        "<I,I>",
    ]

    # Set up the figure and axes if not provided
    if fig is None and axs is None:
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # Loop through the columns in the data array
    for i in range(9):
        # Clear the current axis
        axs[i // 3, i % 3].cla()

        # Get the current column data
        col_data = data_q1[:, i]
        col_data1 = data_q2[:, i]

        # Plot the data on the current axis
        axs[i // 3, i % 3].plot(xaxis, col_data)
        axs[i // 3, i % 3].plot(xaxis, col_data1)

        # Set the x-axis label
        axs[i // 3, i % 3].set_xlabel("CR time [ns]")

        # Set the y-axis label
        axs[i // 3, i % 3].set_ylabel(col_titles[i])

    # Pause for 0.1 seconds
    fig.suptitle("CR power rabi two qubit QST")
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)


def play_flattop(cr: str, duration: int, sign: str):
    """
    QUA macro to play a gapless flat_top gaussian
    """
    if sign == "positive":
        wait(17, cr + "_twin")
        play("gaussian_rise", cr + "_twin")
        wait(int(rise_fall_length // 4), cr)
        play("flat_top", cr, duration=duration)
        wait(duration, cr + "_twin")
        play("gaussian_fall", cr + "_twin")
    elif sign == "negative":
        wait(17, cr + "_twin")
        play("gaussian_rise" * amp(-1), cr + "_twin")
        wait(int(rise_fall_length // 4), cr)
        play("flat_top" * amp(-1), cr, duration=duration)
        wait(duration, cr + "_twin")
        play("gaussian_fall" * amp(-1), cr + "_twin")


###################
# The QUA program #
###################
times = np.arange(4, 200, 2)  # In clock cycles = 4ns
cooldown_time = 1 * u.us
n_avg = 1000

with program() as CR_time_rabi_one_qst:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    f = declare(int)
    t = declare(int)
    c = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, times)):
            with for_(c, 0, c < 9, c + 1):
                # |0>+|1> (superposition) control - CR
                play("x90", "q1_xy")
                align()
                play_flattop("cr_c1t2", duration=t, sign="positive")
                align()
                play("x180", "q1_xy")
                align()
                play_flattop("cr_c1t2", duration=t, sign="negative")
                align()
                play("x180", "q1_xy")
                align()
                two_qb_QST("q1_xy", "q2_xy", pi_len, pi_len)
                align()
                # Start using Rotated-Readout:
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(9).buffer(len(times)).average().save("I1")
        Q_st[0].buffer(9).buffer(len(times)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(9).buffer(len(times)).average().save("I2")
        Q_st[1].buffer(9).buffer(len(times)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

simulate = True
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, CR_time_rabi_one_qst, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(CR_time_rabi_one_qst)

    fig, ax = plt.subplots(2, 2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plot_tomography_results(I1, I2, times * 4)
    # Close the quantum machines at the end
    qm.close()

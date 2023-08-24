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


def one_qb_QST(qb: str, len: float):
    """
    QUA macro to do single qubit quantum state tomography
    """
    with switch_(c):
        with case_(0):  # projection along X
            play("-y90", qb)
        with case_(1):  # projection along Y
            play("x90", qb)
        with case_(2):  # projection along Z
            wait(len * u.ns, qb)


def plot_tomography_results(array, xaxis, fig=None, axs=None):
    """
    Helper function to display quantum state tomography data
    """
    if fig is None and axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    axs = axs.ravel()
    for i in range(3):
        axs[i].cla()
        axs[i].plot(xaxis, array[:, i, 0], label="control |0>")
        axs[i].plot(xaxis, array[:, i, 1], label="control |0>")
        axs[i].set_title(f"<{chr(88 + i)}>")
        axs[i].set_xlabel("CR length [ns]")
        axs[i].set_ylabel("State probability")
    # axs[3].cla()
    # axs[3].plot(xaxis, get_r_vector(array), label='Data dimension 0')
    # axs[3].set_xlabel("CR length [ns]")
    # axs[3].set_ylabel("R-vector")
    plt.tight_layout()
    plt.pause(0.1)
    plt.show()


###################
# The QUA program #
###################
amplitudes = np.arange(0.0, 1.9, 0.1)
cooldown_time = 1 * u.us
n_avg = 1000

with program() as CR_power_rabi_one_qst:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    f = declare(int)
    a = declare(fixed)
    c = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amplitudes)):
            with for_(c, 0, c < 3, c + 1):
                # |0> control - CR
                play("square_positive" * amp(a), "cr_c1t2")
                align()
                one_qb_QST("q2_xy", pi_len)
                align()
                # Start using Rotated-Readout:
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                wait(cooldown_time * u.ns)

                align()  # global align

                # |1> control - CR
                play("x180", "q1_xy")
                align()
                play("square_positive" * amp(a), "cr_c1t2")
                align()
                one_qb_QST("q2_xy", pi_len)
                align()
                # Start using Rotated-Readout:
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("I1")
        Q_st[0].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("I2")
        Q_st[1].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

simulate = True
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, CR_power_rabi_one_qst, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(CR_power_rabi_one_qst)

    fig, ax = plt.subplots(2, 2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plot_tomography_results(I2, amplitudes)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

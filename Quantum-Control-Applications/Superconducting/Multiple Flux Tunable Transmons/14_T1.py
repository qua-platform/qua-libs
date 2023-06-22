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


###################
# The QUA program #
###################
t_delay = np.arange(4, 3500, 40)
cooldown_time = 1 * u.us
n_avg = 1000

# QUA program
with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(t, t_delay)):
            # qubit 1
            play("x180", "q1_xy")
            wait(t, "q1_xy")

            # qubit 2
            play("x180", "q2_xy")
            wait(t, "q2_xy")

            align()
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
            wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(t_delay)).average().save("I1")
        Q_st[0].buffer(len(t_delay)).average().save("Q1")
        # resonator
        I_st[1].buffer(len(t_delay)).average().save("I2")
        Q_st[1].buffer(len(t_delay)).average().save("Q2")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, T1, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(T1)
    fig, ax = plt.subplots(2,2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        ax[0, 0].cla()
        ax[0, 0].plot(4 * t_delay, I1)
        ax[0, 0].set_title("I1")
        ax[1, 0].cla()
        ax[1, 0].plot(4 * t_delay, Q1)
        ax[1, 0].set_title("Q1")
        ax[1, 0].set_xlabel("Wait time (ns)")
        ax[1, 0].get_shared_x_axes().join(ax[1,0], ax[0,0])
        ax[0, 1].cla()
        ax[0, 1].plot(4 * t_delay, I2)
        ax[0, 1].set_title("I2")
        ax[1, 1].cla()
        ax[1, 1].plot(4 * t_delay, Q2)
        ax[1, 1].set_title("Q2")
        ax[1, 1].get_shared_x_axes().join(ax[1,1], ax[0,1])
        ax[1, 1].set_xlabel("Wait time (ns)")
        plt.tight_layout()
        plt.pause(0.1)

plt.show()
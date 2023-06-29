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
times = np.arange(4, 200, 2)  # In clock cycles = 4ns
cooldown_time = 1 * u.us
n_avg = 1000

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    f = declare(int)
    t = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, times)):
            play("x180", "q1_xy", duration=t)
            # play("x180", "q2_xy", duration=t*u.ns)
            align()

            # Start using Rotated-Readout:
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
            wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(times)).average().save("I1")
        Q_st[0].buffer(len(times)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(times)).average().save("I2")
        Q_st[1].buffer(len(times)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, rabi, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi)

    fig, ax = plt.subplots(2, 2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(221)
        plt.cla()
        plt.plot(times, I1)
        plt.title("I1")
        plt.subplot(223)
        plt.cla()
        plt.plot(times, Q1)
        plt.title("Q1")
        plt.xlabel("qubit pulse duration (ns)")
        plt.subplot(222)
        plt.cla()
        plt.plot(times, I2)
        plt.title("I2")
        plt.subplot(224)
        plt.cla()
        plt.plot(times, Q2)
        plt.title("Q2")
        plt.xlabel("qubit pulse duration (ns)")
        plt.tight_layout()
        plt.pause(1.0)

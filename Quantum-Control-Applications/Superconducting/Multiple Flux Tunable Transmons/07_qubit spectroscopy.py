from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm.simulate import LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout


t = 14000   # Qubit pulse length

dfs = np.arange(- 20e6, + 20e6, 0.1e6)
n_avg = 10000

cooldown_time = 1 * u.us

###################
# The QUA program #
###################
with program() as multi_qubit_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            update_frequency("q1_xy", df + qubit_IF_q1)
            update_frequency("q2_xy", df + qubit_IF_q2)

            # qubit 1
            play("cw" * amp(0.07), "q1_xy", duration=t * u.ns)
            align("q1_xy", "rr1")
            # qubit 2
            play("cw" * amp(0.3), "q2_xy", duration=t * u.ns)
            align("q2_xy", "rr2")

            # readout (reduce amplitude to minimize measurement induced transitions)
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], amplitude=0.9)

            wait(cooldown_time * u.ns)


    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dfs)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = True
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, multi_qubit_spec, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(multi_qubit_spec)

    fig, ax = plt.subplots(2, 2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)
        s1 = u.demod2volts(I1 + 1j * Q1, readout_len)
        s2 = u.demod2volts(I2 + 1j * Q2, readout_len)

        ax[0, 0].cla()
        ax[1, 0].cla()
        ax[0, 1].cla()
        ax[1, 1].cla()
        ax[0, 0].plot(dfs / u.MHz, np.abs(s1))
        ax[0, 0].set_title(f"q1 amp (fcent1: {(qubit_LO + qubit_IF_q1) / u.MHz} MHz)")
        ax[1, 0].plot(dfs / u.MHz, np.angle(s1))
        ax[0, 1].plot(dfs / u.MHz, np.abs(s2))
        ax[0, 1].set_title(f"q2 amp (fcent2: {(qubit_LO + qubit_IF_q2) / u.MHz} MHz)")
        ax[1, 1].plot(dfs / u.MHz, np.angle(s2))
        ax[1, 0].set_ylabel("phase (rad)")
        ax[0, 0].set_ylabel("amplitude (V)")
        ax[1, 1].set_xlabel("detuning (MHz)")
        ax[1, 0].set_xlabel("detuning (MHz)")
        plt.tight_layout()
        plt.pause(0.1)

    # plt.plot(I1, Q1, '.')
    # plt.plot(I2, Q2, '.')
    # plt.axis('equal')

plt.show()
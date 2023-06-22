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
amps = np.arange(0.1, 1.7, 0.01)
cooldown_time = 1 * u.us
n_avg = 1000
err_amp = 10

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    f = declare(int)
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amps)):
            # Loop for error amplification (perform many qubit pulses)
            for i in range(err_amp):
                play("x180" * amp(a), "q1_xy")
                # play("x180" * amp(a), "q2_xy")
            align()

            # Start using Rotated-Readout:
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
            wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).average().save("I1")
        Q_st[0].buffer(len(amps)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).average().save("I2")
        Q_st[1].buffer(len(amps)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = True
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
        # s1 = I1 + 1j*Q1
        # s2 = I2 + 1j*Q2

        u = unit()
        ax[0, 0].cla()
        ax[0, 0].plot(amps, I1)
        ax[0, 0].set_title("I1")
        ax[1, 0].cla()
        ax[1, 0].plot(amps, Q1)
        ax[1, 0].set_title("Q1")
        ax[1, 0].set_xlabel("qubit pulse amplitude prefactor (V)")
        ax[0, 1].cla()
        ax[0, 1].plot(amps, I2)
        ax[0, 1].set_title("I2")
        ax[1, 1].cla()
        ax[1, 1].plot(amps, Q2)
        ax[1, 1].set_title("Q2")
        ax[1, 1].set_xlabel("qubit pulse amplitude prefactor (V)")
        plt.tight_layout()
        plt.pause(1.0)

    # plt.plot(I1, Q1, '.')
    # plt.plot(I2, Q2, '.')
    # plt.axis('equal')

plt.show()
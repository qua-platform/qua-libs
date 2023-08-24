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
N_pi = 10
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")[::2]

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    f = declare(int)
    a = declare(fixed)
    npi = declare(int)
    count = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(npi, N_pi_vec)):
            with for_(*from_array(a, amps)):
                # Loop for error amplification (perform many qubit pulses)
                with for_(count, 0, count < npi, count + 1):
                    play("x180" * amp(a), "q1_xy")
                    # play("x180" * amp(a), "q2_xy")
                align()

                # Start using Rotated-Readout:
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, rabi, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi)

    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        if I1.shape[0] > 1:
            plt.subplot(221)
            plt.cla()
            plt.pcolor(amps, N_pi_vec, I1)
            plt.title("I1")
            plt.subplot(223)
            plt.cla()
            plt.pcolor(amps, N_pi_vec, Q1)
            plt.title("Q1")
            plt.xlabel("qubit pulse amplitude pre-factor (V)")
            plt.ylabel("Number of pi pulses")
            plt.subplot(222)
            plt.cla()
            plt.pcolor(amps, N_pi_vec, I2)
            plt.title("I2")
            plt.subplot(224)
            plt.cla()
            plt.pcolor(amps, N_pi_vec, Q2)
            plt.title("Q2")
            plt.xlabel("qubit pulse amplitude pre-factor (V)")
            plt.ylabel("Number of pi pulses")
        else:
            plt.subplot(221)
            plt.cla()
            plt.plot(amps, I1[0])
            plt.title("I1")
            plt.subplot(223)
            plt.cla()
            plt.plot(amps, Q1[0])
            plt.title("Q1")
            plt.xlabel("qubit pulse amplitude pre-factor (V)")
            plt.subplot(222)
            plt.cla()
            plt.plot(amps, I2[0])
            plt.title("I2")
            plt.subplot(224)
            plt.cla()
            plt.plot(amps, Q2[0])
            plt.title("Q2")
            plt.xlabel("qubit pulse amplitude pre-factor (V)")
        plt.tight_layout()
        plt.pause(1.0)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

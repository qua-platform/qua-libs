from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout
from quam import QuAM
from configuration import build_config, u

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
dfs = np.arange(-10e6, 10e6, 0.1e6)
t_delay = np.arange(4, 300, 4)
n_avg = 1000
cooldown_time = 1 * u.us

qb_if_1 = machine.qubits[0].xy.f_01 - machine.local_oscillators.qubits[0].freq
qb_if_2 = machine.qubits[1].xy.f_01 - machine.local_oscillators.qubits[0].freq

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)
    df = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency("q0_xy", df + qb_if_1)
            update_frequency("q1_xy", df + qb_if_2)

            with for_(*from_array(t, t_delay)):
                # qubit 1
                play("x90", "q0_xy")
                wait(t, "q0_xy")
                play("x90", "q0_xy")

                # qubit 2
                play("x90", "q1_xy")
                wait(t, "q1_xy")
                play("x90", "q1_xy")

                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1], weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(t_delay)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(t_delay)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(t_delay)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(t_delay)).buffer(len(dfs)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulate = False
if simulate:
    job = qmm.simulate(config, ramsey, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(ramsey)

    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(221)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, I1)
        plt.title(f"Q1-I, fcent={machine.qubits[0].xy.f_01 / u.MHz}")
        plt.ylabel("detuning (MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, Q1)
        plt.title("Q1-Q")
        plt.xlabel("Idle time (ns)")
        plt.ylabel("detuning (MHz)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, I2)
        plt.title(f"Q2-I, fcent={machine.qubits[1].xy.f_01 / u.MHz}")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, Q2)
        plt.title("Q2-Q")
        plt.xlabel("Idle time (ns)")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

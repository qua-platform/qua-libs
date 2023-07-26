from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout
from qualang_tools.plot.fitting import Fit
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
idle_times = np.arange(4, 300, 1)
cooldown_time = 1 * u.us
n_avg = 1000
detuning = 1e6
qubit_element = "q0_xy"

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)
    phi = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(t, idle_times)):
            # play("x180", "q1_xy")
            # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
            assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
            play("x90", qubit_element)
            wait(t, "q0_xy")
            frame_rotation_2pi(phi, qubit_element)
            play("x90", qubit_element)

            align()
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1], weights="rotated_")
            reset_frame(qubit_element)
            wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(idle_times)).average().save("I1")
        Q_st[0].buffer(len(idle_times)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(idle_times)).average().save("I2")
        Q_st[1].buffer(len(idle_times)).average().save("Q2")

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

    fig, ax = plt.subplots(2, 2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")

    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(221)
        plt.cla()
        plt.plot(4 * idle_times, I1)
        plt.title("I1")
        plt.subplot(223)
        plt.cla()
        plt.plot(4 * idle_times, Q1)
        plt.title("Q1")
        plt.xlabel("idle_times (ns)")
        plt.subplot(222)
        plt.cla()
        plt.plot(4 * idle_times, I2)
        plt.title("I2")
        plt.subplot(224)
        plt.cla()
        plt.plot(4 * idle_times, Q2)
        plt.title("Q2")
        plt.xlabel("idle_times (ns)")
        plt.tight_layout()
        plt.pause(0.1)

try:
    fit = Fit()
    plt.figure()
    plt.subplot(221)
    fit_1 = fit.ramsey(4 * idle_times, I1, plot=True)
    plt.xlabel("idle_times (ns)")
    plt.subplot(223)
    fit_1 = fit.ramsey(4 * idle_times, Q1, plot=True)
    plt.xlabel("idle_times (ns)")
    plt.subplot(222)
    fit_2 = fit.ramsey(4 * idle_times, I2, plot=True)
    plt.xlabel("idle_times (ns)")
    plt.subplot(224)
    fit_2 = fit.ramsey(4 * idle_times, Q2, plot=True)
    plt.xlabel("idle_times (ns)")
except (Exception,):
    pass

# machine.qubits[0].T2 = fit_1["tau"]
# machine.qubits[0].xy.f_01 += detuning - fit_1["f"] * 1e6
# machine.qubits[1].T2 = fit_2["tau"]
# machine.qubits[1].xy.f_01 += detuning - fit_2["f"] * 1e6
# machine._save("quam_bootstrap_state.json")

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

from qm.simulate import LoopbackInterface
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

###################
# The QUA program #
###################
dfs = np.arange(-14e6, +14e6, 0.2e6)
amps = np.arange(0.0, 1, 0.02)

cooldown_time = 1 * u.us
n_avg = 1000

qb_if_1 = machine.qubits[0].xy.f_01 - machine.local_oscillators.qubits[0].freq
qb_if_2 = machine.qubits[1].xy.f_01 - machine.local_oscillators.qubits[0].freq

with program() as rabi_chevron:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)
    f_q1 = declare(int)
    f_q2 = declare(int)
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency("q0_xy", df + qb_if_1)
            update_frequency("q1_xy", df + qb_if_2)

            with for_(*from_array(a, amps)):
                # qubit 1 can replace cw by x180 to test the gate
                play("x180" * amp(a), "q0_xy")
                play("x180" * amp(a), "q1_xy")
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1])
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("Q2")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        rabi_chevron,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi_chevron)

    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        s1 = u.demod2volts(I1 + 1j * Q1, machine.resonators[0].readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, machine.resonators[0].readout_pulse_length)

        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * machine.qubits[0].xy.pi_amp, dfs, I1)
        plt.xlabel("qubit pulse amplitude (V)")
        plt.ylabel("qubit 1 detuning (MHz)")
        plt.title(f"q1 (f_res1: {machine.qubits[0].xy.f_01 / u.MHz} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * machine.qubits[0].xy.pi_amp, dfs, Q1)
        plt.xlabel("qubit pulse amplitude (V)")
        plt.ylabel("qubit 1 detuning (MHz)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * machine.qubits[1].xy.pi_amp, dfs, I2)
        plt.title(f"q2 (f_res2: {machine.qubits[1].xy.f_01 / u.MHz} MHz)")
        plt.ylabel("qubit 2 detuning (MHz)")
        plt.xlabel("qubit pulse amplitude (V)")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * machine.qubits[1].xy.pi_amp, dfs, Q2)
        plt.xlabel("qubit pulse amplitude (V)")
        plt.ylabel("qubit 2 detuning (MHz)")
        plt.tight_layout()
        plt.pause(0.1)

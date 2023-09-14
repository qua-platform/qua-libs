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
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)
config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

###################

###################
# The QUA program #
###################
dfs = np.arange(-100e6, +100e6, 1e6)
amps = np.arange(0.0, 1.9, 0.02)

cooldown_time = 5 * max(qb1.T1, qb2.T1)
n_avg = 100


with program() as rabi_chevron:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)
    f_q1 = declare(int)
    f_q2 = declare(int)
    a = declare(fixed)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency(qb1.name + "_xy", df + qb_if_1)
            update_frequency(qb2.name + "_xy", df + qb_if_2)

            with for_(*from_array(a, amps)):
                # qubit 1 can replace cw by x180 to test the gate
                play("x180" * amp(a), qb1.name + "_xy")
                play("x180" * amp(a), qb2.name + "_xy")
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits)
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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

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

        I1, Q1 = u.demod2volts(I1, rr1.readout_pulse_length), u.demod2volts(Q1, rr1.readout_pulse_length)
        I2, Q2 = u.demod2volts(I2, rr2.readout_pulse_length), u.demod2volts(Q2, rr2.readout_pulse_length)

        plt.suptitle("Rabi chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * qb1.xy.pi_amp, dfs / u.MHz, I1)
        plt.plot(qb1.xy.pi_amp, 0, "r*")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.title(f"{qb1.name} (f_res1: {int((qb_if_1 + lo1) / u.MHz)} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * qb1.xy.pi_amp, dfs / u.MHz, Q1)
        plt.plot(qb1.xy.pi_amp, 0, "r*")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * qb2.xy.pi_amp, dfs / u.MHz, I2)
        plt.plot(qb2.xy.pi_amp, 0, "r*")
        plt.title(f"{qb2.name} (f_res2: {int((qb_if_2 + lo2) / u.MHz)} MHz)")
        plt.ylabel("Qubit detuning [MHz]")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * qb2.xy.pi_amp, dfs / u.MHz, Q2)
        plt.plot(qb2.xy.pi_amp, 0, "r*")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.tight_layout()
        plt.pause(5)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

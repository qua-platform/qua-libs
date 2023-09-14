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
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
###################
# The QUA program #
###################
times = np.arange(4, 200, 2)  # In clock cycles = 4ns
cooldown_time = 5 * max(qb1.T1, qb2.T1)
n_avg = 1000

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    f = declare(int)
    t = declare(int)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, times)):
            play("x180", qb1.name + "_xy", duration=t)
            play("x180", qb2.name + "_xy", duration=t)
            align()

            # Start using Rotated-Readout:
            multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

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

        I1, Q1 = u.demod2volts(I1, rr1.readout_pulse_length), u.demod2volts(Q1, rr1.readout_pulse_length)
        I2, Q2 = u.demod2volts(I2, rr2.readout_pulse_length), u.demod2volts(Q2, rr2.readout_pulse_length)

        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("Time Rabi")
        plt.subplot(221)
        plt.cla()
        plt.plot(times * 4, I1)
        plt.title(f"{qb1.name}")
        plt.ylabel("I quadrature [V]")
        plt.subplot(223)
        plt.cla()
        plt.plot(times * 4, Q1)
        plt.xlabel("qubit pulse duration [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.subplot(222)
        plt.cla()
        plt.plot(times * 4, I2)
        plt.title(f"{qb2.name}")
        plt.subplot(224)
        plt.cla()
        plt.plot(times * 4, Q2)
        plt.xlabel("qubit pulse duration [ns]")
        plt.tight_layout()
        plt.pause(1.0)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

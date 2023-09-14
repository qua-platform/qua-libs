#%%
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
# The QUA program #
###################
t = 40 * u.ns  # Qubit pulse length
cooldown_time = 5 * max(qb1.T1, qb2.T1)

dfs = np.arange(-60e6, +80e6, 1e6)
n_avg = 300


with program() as multi_qubit_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)
    
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            update_frequency(qb1.name + "_xy", df + qb_if_1)
            update_frequency(qb2.name + "_xy", df + qb_if_2)

            # qubit 1
            play("x180" * amp(1), qb1.name + "_xy", duration=t * u.ns)
            align(qb1.name + "_xy", rr1.name)
            # qubit 2
            play("x180" * amp(1), qb2.name + "_xy", duration=t * u.ns)
            align(qb2.name + "_xy", rr2.name)

            # readout (reduce amplitude to minimize measurement induced transitions)
            multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, amplitude=0.9)

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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, multi_qubit_spec, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # execute QUA:
    qm = qmm.open_qm(config)
    job = qm.execute(multi_qubit_spec)

    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)
        s1 = u.demod2volts(I1 + 1j * Q1, rr1.readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, rr2.readout_pulse_length)
        
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(221)
        plt.cla()
        plt.plot(dfs / u.MHz, np.abs(s1))
        plt.grid("on")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{qb1.name} (f_01: {qb1.xy.f_01 / u.MHz} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.plot(dfs / u.MHz, np.angle(s1))
        plt.grid("on")
        plt.ylabel("Phase [rad]")
        plt.xlabel(f"{qb1.name} detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.plot(dfs / u.MHz, np.abs(s2))
        plt.grid("on")
        plt.title(f"{qb2.name} (f_01: {qb2.xy.f_01 / u.MHz} MHz)")
        plt.subplot(224)
        plt.cla()
        plt.plot(dfs / u.MHz, np.angle(s2))
        plt.grid("on")
        plt.xlabel(f"{qb1.name} detuning [MHz]")
        plt.tight_layout()
        plt.pause(2)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(121)
        res_1 = fit.reflection_resonator_spectroscopy((qb_if_1 + dfs) / u.MHz, -np.angle(s1), plot=True)
        plt.legend((f"f = {res_1['f'][0]:.3f} MHz",))
        plt.xlabel(f"{rr1.name} IF [MHz]")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{qb1.name}")
        plt.subplot(122)
        res_2 = fit.reflection_resonator_spectroscopy((qb_if_2 + dfs) / u.MHz, np.abs(s2), plot=True)
        plt.legend((f"f = {res_2['f'][0]:.3f} MHz",))
        plt.xlabel(f"{rr2.name} IF [MHz]")
        plt.title(f"{qb2.name}")
        plt.tight_layout()

        qb1.xy.f_01 = res_1["f"][0] * u.MHz + lo1
        qb2.xy.f_01 = res_2["f"][0] * u.MHz + lo2
    except (Exception,):
        pass
# qb1.xy.f_01 =
# qb2.xy.f_01 =
# machine._save("current_state.json")

# %%

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
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
ts = np.arange(4, 200, 1)
dcs = np.arange(-0.105, -0.10, 0.0001)

cooldown_time = 5 * max(qb1.T1, qb2.T1)
n_avg = 40

with program() as iswap:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)
    dc = declare(fixed)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, ts)):
            with for_(*from_array(dc, dcs)):
                play("x180", qb1.name + "_xy")
                align()
                set_dc_offset(q2_z, "single", dc)
                wait(t, q2_z)
                align()
                set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)
                wait(100)
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(ts)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(ts)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(ts)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(ts)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False
if simulate:
    job = qmm.simulate(config, iswap, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(iswap)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("iSWAP chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I1)
        plt.title(f"{qb1.name} - I, f_01={int(qb1.xy.f_01 / u.MHz)} MHz")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q1)
        plt.title(f"{qb1.name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I2)
        plt.title(f"{qb2.name} - I, f_01={int(qb2.xy.f_01 / u.MHz)} MHz")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q2)
        plt.title(f"{qb2.name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.tight_layout()
        plt.pause(1)

    # np.savez(save_dir / 'iswap', I1=I1, Q1=Q1, I2=I2, ts=ts, dcs=dcs)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
# qb1.z.iswap.length =
# qb1.z.iswap.level =
# machine._save("current_state.json")

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
from configuration import build_config, u

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)


###################
# The QUA program #
###################
ts = np.arange(4, 200, 1)
dcs = np.arange(-0.315, -0.298, 0.0002)
cooldown_time = 1 * u.us
n_avg = 1300

with program() as iswap:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)
    dc = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, ts)):
            with for_(*from_array(dc, dcs)):
                play("x180", "q1_xy")
                align()
                set_dc_offset("q0_z", "single", dc)
                wait(t, "q0_z")
                align()
                set_dc_offset("q0_z", "single", machine.qubits[0].z.max_frequency_point)
                wait(10)
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1], weights="rotated_")
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
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulate = True
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

        plt.subplot(221)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I1)
        plt.title("q1 - I")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q1)
        plt.title("q1 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, I2)
        plt.title("q2 - I")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(dcs, 4 * ts, Q2)
        plt.title("q2 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.tight_layout()
        plt.pause(0.1)
    # np.savez(save_dir / 'iswap', I1=I1, Q1=Q1, I2=I2, ts=ts, dcs=dcs)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
# machine.qubits[0].z.iswap.length =
# machine.qubits[0].z.iswap.level =
# machine._save("quam_bootstrap_state.json")

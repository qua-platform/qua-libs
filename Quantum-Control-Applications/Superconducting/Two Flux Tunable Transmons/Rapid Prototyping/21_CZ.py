from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout


###################
# The QUA program #
###################
dc0_q2 = config["controllers"]["con1"]["analog_outputs"][8]["offset"]
dc0_q1 = config["controllers"]["con1"]["analog_outputs"][7]["offset"]
ts = np.arange(4, 200, 1)
amps = np.arange(-0.098, -0.118, -0.0005)
cooldown_time = 1 * u.us
n_avg = 13000

with program() as cz:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, ts)):
            with for_(*from_array(a, amps)):
                play("x180", "q1_xy")
                play("x180", "q2_xy")
                align()
                set_dc_offset("q1_z", "single", a)
                wait(t, "q1_z")
                align()
                set_dc_offset("q1_z", "single", dc0_q1)
                wait(10)
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save('n')
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(len(ts)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(len(ts)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(len(ts)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(len(ts)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = True
if simulate:
    job = qmm.simulate(config, cz, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cz)
    fig, ax = plt.subplots(2, 2)
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps, 4*ts, I1)
        plt.title('q1 - I')
        plt.ylabel("Interaction time (ns)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps, 4*ts, Q1)
        plt.title('q1 - Q')
        plt.xlabel("FLux amplitude (V)")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps, 4*ts, I2)
        plt.title('q2 - I')
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps, 4*ts, Q2)
        plt.title('q2 - Q')
        plt.xlabel("FLux amplitude (V)")
        plt.tight_layout()
        plt.pause(0.1)

plt.show()
# np.savez(save_dir/'cz', I1=I1, Q1=Q1, I2=I2, Q2=Q2, ts=ts, amps=amps)
"""
Measure the qubit in the ground and excited state to create the IQ blobs. If the separation and the fidelity are good
enough, gives the parameters needed for active reset
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
from TwoStateDiscriminator import TwoStateDiscriminator
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

###################
# The QUA program #
###################

n_runs = 1000

lsb = False

rr_qe = "resonator"

cooldown_time = 5 * qubit_T1


discriminator = TwoStateDiscriminator(
    qmm=qmm,
    config=config,
    update_tof=False,
    rr_qe=rr_qe,
    path=f"ge_disc_params_{rr_qe}.npz",
    lsb=lsb,
    meas_len=readout_len,
    smearing=smearing,
)

with program() as benchmark:
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    res = declare(bool)
    I_st = declare_stream()
    Q_st = declare_stream()
    res_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_runs, n + 1):
        discriminator.measure_state("readout", "out1", "out2", res=res, I=I, Q=Q)
        save(I, I_st)
        save(Q, Q_st)
        save(res, res_st)
        wait(cooldown_time * u.ns, "resonator")

        align()  # global align

        play("x180", "qubit")
        align("qubit", "resonator")
        discriminator.measure_state("readout", "out1", "out2", res=res, I=I, Q=Q)
        save(I, I_st)
        save(Q, Q_st)
        save(res, res_st)
        wait(cooldown_time * u.ns, "resonator")

        seq0 = [0, 1] * n_runs

    with stream_processing():
        res_st.boolean_to_int().save_all("res")
        I_st.save_all("I")
        Q_st.save_all("Q")

qm = qmm.open_qm(config)

job = qm.execute(benchmark)

result_handles = job.result_handles
result_handles.wait_for_all_values()
res = result_handles.get("res").fetch_all()["value"]
I = result_handles.get("I").fetch_all()["value"]
Q = result_handles.get("Q").fetch_all()["value"]

plt.figure()
plt.hist(I[np.array(seq0) == 0], 50)
plt.hist(I[np.array(seq0) == 1], 50)
plt.plot([discriminator.get_threshold()] * 2, [0, 60], "g")
plt.show()

plt.figure()
plt.plot(I, Q, ".")
discriminator.plot_sigma_mu()
plt.axis("equal")

p_s = np.zeros(shape=(2, 2))
for i in range(2):
    res_i = res[np.array(seq0) == i]
    p_s[i, :] = np.array([np.mean(res_i == 0), np.mean(res_i == 1)])

labels = ["g", "e"]
plt.figure()
ax = plt.subplot()
sns.heatmap(p_s, annot=True, ax=ax, fmt="g", cmap="Blues")

ax.set_xlabel("Prepared")
ax.set_ylabel("Measured")
ax.set_title("Fidelities")
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)

plt.show()

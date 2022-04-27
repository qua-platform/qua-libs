from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

n_runs = 10000

cooldown_time = 5 * qubit_T1 // 4

with program() as IQ_blobs:
    n = declare(int)
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()

    with for_(n, 0, n < n_runs, n + 1):
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
        )
        save(I_g, I_g_st)
        save(Q_g, Q_g_st)
        wait(cooldown_time, "resonator")

        align()
        play("pi", "qubit")
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
        )
        save(I_e, I_e_st)
        save(Q_e, Q_e_st)
        wait(cooldown_time, "resonator")

        # Assume we have two blobs, we can use the integration weights to rotate them such that all of the information
        # will be in the I axis.
        # See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
        # Once we do this, we can perform active reset using:
        #########################################
        #
        # # Active reset:
        # with if_(I < 0.2):
        #     play("pi", "qubit")
        #
        #########################################
        #
        # # Active reset (faster):
        # play("pi", "qubit", condition=I < 0.2)
        #
        #########################################
        #
        # # Repeat until success active reset
        # with while_(I < 0.2):
        #     play("pi", "qubit")
        #     align("qubit", "resonator")
        #     measure("readout", "resonator", None,
        #                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
        #
        #########################################
        #
        # # Repeat until success active reset, up to 3 iterations
        # count = declare(int)
        # cont_condition = declare(bool)
        # assign(cont_condition, ((I < 0.2) & (count < 3)))
        # with while_(cont_condition):
        #     play("pi", "qubit")
        #     align("qubit", "resonator")
        #     measure("readout", "resonator", None,
        #                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
        #     assign(count, count + 1)
        #
        #########################################

    with stream_processing():
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(IQ_blobs)
res_handles = job.result_handles
res_handles.wait_for_all_values()
I_e = res_handles.get("I_g").fetch_all()["value"]
I_g = res_handles.get("I_e").fetch_all()["value"]
Q_g = res_handles.get("Q_g").fetch_all()["value"]
Q_e = res_handles.get("Q_e").fetch_all()["value"]

plt.figure()
plt.plot(I_g, Q_g, ".", alpha=0.1)
plt.plot(I_e, Q_e, ".", alpha=0.1)
plt.axis("equal")
plt.legend(["Ground", "Excited"])

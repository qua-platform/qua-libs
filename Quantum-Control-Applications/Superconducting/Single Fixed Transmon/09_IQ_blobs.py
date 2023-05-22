"""
IQ_blobs.py: Measure the qubit in the ground and excited state to create the IQ blobs.
If the separation and the fidelity are good enough, gives the parameters needed for active reset
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from qualang_tools.analysis.discriminator import two_state_discriminator

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

        align()  # global align

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

    with stream_processing():
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

qm = qmm.open_qm(config)

job = qm.execute(IQ_blobs)
res_handles = job.result_handles
res_handles.wait_for_all_values()
Ig = res_handles.get("I_g").fetch_all()["value"]
Qg = res_handles.get("Q_g").fetch_all()["value"]
Ie = res_handles.get("I_e").fetch_all()["value"]
Qe = res_handles.get("Q_e").fetch_all()["value"]

angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(Ig, Qg, Ie, Qe, b_print=True, b_plot=True)

#########################################
# The two_state_discriminator gives us the rotation angle which makes it such that all of the information will be in
# the I axis. This is being done by setting the `rotation_angle` parameter in the configuration.
# See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
# Once we do this, we can perform active reset using:
#########################################
#
# # Active reset:
# with if_(I > threshold):
#     play("pi", "qubit")
#
#########################################
#
# # Active reset (faster):
# play("pi", "qubit", condition=I > threshold)
#
#########################################
#
# # Repeat until success active reset
# with while_(I > threshold):
#     play("pi", "qubit")
#     align("qubit", "resonator")
#     measure("readout", "resonator", None,
#                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
#
#########################################
#
# # Repeat until success active reset, up to 3 iterations
# count = declare(int)
# assign(count, 0)
# cont_condition = declare(bool)
# assign(cont_condition, ((I > threshold) & (count < 3)))
# with while_(cont_condition):
#     play("pi", "qubit")
#     align("qubit", "resonator")
#     measure("readout", "resonator", None,
#                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
#     assign(count, count + 1)
#     assign(cont_condition, ((I > threshold) & (count < 3)))
#
#########################################

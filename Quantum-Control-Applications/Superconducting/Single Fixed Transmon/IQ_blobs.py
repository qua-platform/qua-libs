from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from scipy.optimize import minimize

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
Ig = res_handles.get("I_g").fetch_all()["value"]
Ie = res_handles.get("I_e").fetch_all()["value"]
Qg = res_handles.get("Q_g").fetch_all()["value"]
Qe = res_handles.get("Q_e").fetch_all()["value"]

b_print = True
b_plot = True

angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
C = np.cos(angle)
S = np.sin(angle)

Ig_rotated = Ig * C - Qg * S
Qg_rotated = Ig * S + Qg * C

Ie_rotated = Ie * C - Qe * S
Qe_rotated = Ie * S + Qe * C


def false_detections(threshold, Ig, Ie):
    if np.mean(Ig) < np.mean(Ie):
        false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
    else:
        false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
    return false_detections_var


fit = minimize(
    false_detections,
    0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),
    (Ig_rotated, Ie_rotated),
    method="Nelder-Mead",
)
threshold = fit.x[0]

if np.mean(Ig_rotated) < np.mean(Ie_rotated):
    gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
    ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
    eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
    ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)
    threshold_direction_string = "Excited is larger"
else:
    gg = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
    ge = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
    eg = np.sum(Ie_rotated > threshold) / len(Ie_rotated)
    ee = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
    threshold_direction_string = "Ground is larger"

if b_print:
    print(
        f"""
    Fidelity Matrix:
    -----------------
    | {gg:.3f} | {ge:.3f} |
    ----------------
    | {eg:.3f} | {ee:.3f} |
    -----------------
    IQ plane rotated by: {180 / np.pi * angle:.1f}{chr(176)}
    Threshold: {threshold:.3e} ({threshold_direction_string})
    Readout Fidelity: {100*(gg + ee)/2:.1f}%
    """
    )

if b_plot:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(Ig, Qg, ".", alpha=0.1, label='Ground')
    ax1.plot(Ie, Qe, ".", alpha=0.1, label='Excited')
    ax1.axis("equal")
    ax1.legend(["Ground", "Excited"])
    ax1.set_xlabel('I')
    ax1.set_ylabel('Q')
    ax1.set_title('Original Data')

    ax2.plot(Ig_rotated, Qg_rotated, ".", alpha=0.1, label='Ground')
    ax2.plot(Ie_rotated, Qe_rotated, ".", alpha=0.1, label='Excited')
    ax2.axis("equal")
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.set_title('Rotated Data')

    ax3.hist(Ig_rotated, bins=50, alpha=0.75, label='Ground')
    ax3.hist(Ie_rotated, bins=50, alpha=0.75, label='Excited')
    ax3.axvline(x=threshold, color='k', ls='--', alpha=0.5)
    ax3.text(0.7, 0.9, f"{threshold:.3e}", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    ax3.set_xlabel('I')
    ax3.set_title('1D Histogram')

    ax4.imshow(np.array([[gg, ge], [eg, ee]]))
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(labels=['|g>', '|e>'])
    ax4.set_yticklabels(labels=['|g>', '|e>'])
    ax4.set_ylabel('Prepared')
    ax4.set_xlabel('Measured')
    ax4.text(0, 0, f'{100*gg:.1f}%', ha="center", va="center", color="k")
    ax4.text(0, 1, f'{100*eg:.1f}%', ha="center", va="center", color="w")
    ax4.text(1, 0, f'{100*ge:.1f}%', ha="center", va="center", color="w")
    ax4.text(1, 1, f'{100*ee:.1f}%', ha="center", va="center", color="k")
    ax4.set_title('Fidelities')
    fig.tight_layout()

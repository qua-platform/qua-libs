from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from scipy.optimize import minimize

from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

###################
# The QUA program #
###################

# %matplotlib qt

n_runs = 10000
n_cqr = 50

cooldown_time = 5  * qubit_T1 // 4

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
    j = declare(int)
    
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    res_st = declare_stream()
    
    I_th = declare(fixed, value=0.0)
    res = declare(bool)
    
    
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_runs, n + 1):
        
        # update_frequency('squeeze_drive', int(0))
        # update_frequency('squeeze_rise', int(0))
        # update_frequency('cqr_drive', int(0))
        # update_frequency('resonator', int(0))
        
        play('on', 'squeeze_switch', duration=int((4e6+1e3)//4))
        play('ftc_rise', 'squeeze_rise')
        align('squeeze_rise', 'squeeze_drive')
        play('cw', 'squeeze_drive', duration=int(4e6//4))
        align('squeeze_drive', 'squeeze_fall')
        play('ftc_fall', 'squeeze_fall')
        
        wait(cooldown_time, 'cqr_drive', 'resonator')
        # wait(int(10e3//4), 'cqr_drive', 'resonator')
        
        with for_(j, 0, j < n_cqr, j+1):
            play('on', 'cqr_switch', duration=(cqr_len//4))
            play('cqr', 'cqr_drive')
            play('on', 'SPC_pump', duration=(passive_len//4))
            measure('passive_readout', 'resonator', None, demod.full('rotated_cos', I, 'out1'), demod.full('rotated_sin', Q, 'out1'))
            
            # assign(res, Util.cond(I > I_th, True, False))
            assign(res,I>I_th)
            
            # with if_(I>I_th):
            #     assign(res, Util.cond(True, False, I>I_th))
            save(res, res_st)
            
            save(I, I_st)
            save(Q, Q_st)
            # with else_():
            #     assign(res, Util.cond(True,False,I_aux<I_th))
                        
            


        # Assume we have two blobs, we can use the integration weights to rotate them such that all of the information
        # will be in the I axis.
        # See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
        # Once we do this, we can perform active reset using:
        #########################################
        #
        # # Active reset:
        # with if_(I < 0.2):
        #     play("pi", "Fock_qubit")
        #
        #########################################
        #
        # # Active reset (faster):
        # play("pi", "Fock_qubit", condition=I < 0.2)
        #
        #########################################
        #
        # # Repeat until success active reset
        # with while_(I < 0.2):
        #     play("pi", "Fock_qubit")
        #     align("Fock_qubit", "resonator")
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
        #     play("pi", "Fock_qubit")
        #     align("Fock_qubit", "resonator")
        #     measure("readout", "resonator", None,
        #                 dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I))
        #     assign(count, count + 1)
        #
        #########################################

    with stream_processing():
        # adc_st.input1().save_all("adc1")
        I_st.save_all('I')
        Q_st.save_all('Q')
        # I_st.buffer(n_cqr).buffer(n_runs).save('I_traj')
        # res_st.boolean_to_int().save_all('res')
        res_st.boolean_to_int().buffer(n_cqr).buffer(n_runs).save('res')

    #     I_g_st.save_all("I_g")
    #     Q_g_st.save_all("Q_g")
    #     I_e_st.save_all("I_e")
    #     Q_e_st.save_all("Q_e")

#####################################
#  Open Communication with the QOP  #
#####################################
# qmm = QuantumMachinesManager()


qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)
# job = qmm.simulate(config, IQ_blobs, SimulationConfig(duration=int(20e3//4)), flags=['auto-element-thread'])
# job.get_simulated_samples().con1.plot()

qm = qmm.open_qm(config)

job = qm.execute(IQ_blobs)
res_handles = job.result_handles
res_handles.wait_for_all_values()

I = res_handles.get('I').fetch_all()['value']
Q = res_handles.get('Q').fetch_all()['value']
# res = 2*res_handles.get('res').fetch_all()['value'] -1
res = 2*res_handles.get('res').fetch_all() -1
# adc1 = res_handles.get("adc1").fetch_all()['value'] / 2**12


plt.figure()
# plt.plot(I, Q,'o')
# plt.hist2d(I,Q, bins = 50, norm = mpl.colors.LogNorm())
plt.hist2d(I,Q, bins = 50)
plt.axis('equal')

plt.figure()

trj_pts = 50


time = np.linspace(0,4*trj_pts,trj_pts)
#plt.plot(time,I[:trj_pts]/max(I[:trj_pts]), 'o-')
plt.xlabel('time ($\mu s$)')

plt.plot(time,res[:trj_pts])

I_thresh = 0

I_avg_plus = np.zeros(n_cqr)
I_avg_minus = np.zeros(n_cqr)

n_plus = 0 
n_minus = 0
for i in range(n_runs):
    if(res[i][0]>I_thresh):
        n_plus += 1
        I_avg_plus += res[i]
        # plt.plot(time,res[i])
    else:
        n_minus += 1
        I_avg_minus += res[i]

I_avg_plus = I_avg_plus/n_plus
I_avg_minus = I_avg_minus/n_minus       
plt.figure()
plt.plot(time,I_avg_plus,'o-')
plt.plot(time,I_avg_minus,'o-')     
plt.xlabel('time ($\mu s$)')

# plt.figure()
# plt.title("Average7d run (Check ToF & DC Offset)")
# plt.plot(adc1[0])
# plt.show()
# Ig = res_handles.get("I_g").fetch_all()["value"]
# Ie = res_handles.get("I_e").fetch_all()["value"]
# Qg = res_handles.get("Q_g").fetch_all()["value"]
# Qe = res_handles.get("Q_e").fetch_all()["value"]
#
# b_print = True
# b_plot = True
#
# angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
# C = np.cos(angle)
# S = np.sin(angle)
#
# Ig_rotated = Ig * C - Qg * S
# Qg_rotated = Ig * S + Qg * C
#
# Ie_rotated = Ie * C - Qe * S
# Qe_rotated = Ie * S + Qe * C
#
#
# def false_detections(threshold, Ig, Ie):
#     if np.mean(Ig) < np.mean(Ie):
#         false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
#     else:
#         false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
#     return false_detections_var
#
#
# fit = minimize(
#     false_detections,
#     0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),
#     (Ig_rotated, Ie_rotated),
#     method="Nelder-Mead",
# )
# threshold = fit.x[0]
#
# if np.mean(Ig_rotated) < np.mean(Ie_rotated):
#     gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
#     ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
#     eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
#     ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)
#     threshold_direction_string = "Excited is larger"
# else:
#     gg = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
#     ge = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
#     eg = np.sum(Ie_rotated > threshold) / len(Ie_rotated)
#     ee = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
#     threshold_direction_string = "Ground is larger"
#
# if b_print:
#     print(
#         f"""
#     Fidelity Matrix:
#     -----------------
#     | {gg:.3f} | {ge:.3f} |
#     ----------------
#     | {eg:.3f} | {ee:.3f} |
#     -----------------
#     IQ plane rotated by: {180 / np.pi * angle:.1f}{chr(176)}
#     Threshold: {threshold:.3e} ({threshold_direction_string})
#     Readout Fidelity: {100*(gg + ee)/2:.1f}%
#     """
#     )
#
# if b_plot:
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#     ax1.plot(Ig, Qg, ".", alpha=0.1, label="Ground")
#     ax1.plot(Ie, Qe, ".", alpha=0.1, label="Excited")
#     ax1.axis("equal")
#     ax1.legend(["Ground", "Excited"])
#     ax1.set_xlabel("I")
#     ax1.set_ylabel("Q")
#     ax1.set_title("Original Data")
#
#     ax2.plot(Ig_rotated, Qg_rotated, ".", alpha=0.1, label="Ground")
#     ax2.plot(Ie_rotated, Qe_rotated, ".", alpha=0.1, label="Excited")
#     ax2.axis("equal")
#     ax2.set_xlabel("I")
#     ax2.set_ylabel("Q")
#     ax2.set_title("Rotated Data")
#
#     ax3.hist(Ig_rotated, bins=50, alpha=0.75, label="Ground")
#     ax3.hist(Ie_rotated, bins=50, alpha=0.75, label="Excited")
#     ax3.axvline(x=threshold, color="k", ls="--", alpha=0.5)
#     text_props = dict(horizontalalignment="center", verticalalignment="center", transform=ax3.transAxes)
#     ax3.text(0.7, 0.9, f"{threshold:.3e}", text_props)
#     ax3.set_xlabel("I")
#     ax3.set_title("1D Histogram")
#
#     ax4.imshow(np.array([[gg, ge], [eg, ee]]))
#     ax4.set_xticks([0, 1])
#     ax4.set_yticks([0, 1])
#     ax4.set_xticklabels(labels=["|g>", "|e>"])
#     ax4.set_yticklabels(labels=["|g>", "|e>"])
#     ax4.set_ylabel("Prepared")
#     ax4.set_xlabel("Measured")
#     ax4.text(0, 0, f"{100*gg:.1f}%", ha="center", va="center", color="k")
#     ax4.text(0, 1, f"{100*eg:.1f}%", ha="center", va="center", color="w")
#     ax4.text(1, 0, f"{100*ge:.1f}%", ha="center", va="center", color="w")
#     ax4.text(1, 1, f"{100*ee:.1f}%", ha="center", va="center", color="k")
#     ax4.set_title("Fidelities")
#     fig.tight_layout()

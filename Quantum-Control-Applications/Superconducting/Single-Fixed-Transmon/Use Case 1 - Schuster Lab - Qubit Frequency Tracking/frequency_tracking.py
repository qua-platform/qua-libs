from qm import SimulationConfig
from qm.qua import *
from qm import LoopbackInterface
from qm import QuantumMachinesManager
from frequency_tracking_class import qubit_frequency_tracking, qmm
from configuration import config, ge_IF, qubit_freq, disc_file_jpa
import numpy as np

# from qualang_tools.addons.InteractivePlotLib import InteractivePlotLib
import matplotlib.pyplot as plt
import time

# initialize object
freq_track_obj = qubit_frequency_tracking("qubit", "rr", ge_IF)

# time domain ramsey
qm = qmm.open_qm(config)
print(freq_track_obj.fres)
freq_track_obj.f_ref = int(0.06e6)
reps = 20
oscillation = 1
for arg in ["Pe_0", "Pe_corrected"]:
    with program() as prog:
        freq_track_obj.qua_declarations()
        freq_track_obj.time_domain_ramesy_full_sweep(reps, freq_track_obj.f_ref, 4, 50000, 50, arg)

        with stream_processing():
            freq_track_obj.state_estimation_st[0].buffer(reps, len(freq_track_obj.tau_vec)).map(
                FUNCTIONS.average()
            ).save(arg)

    job = qm.execute(prog)
    job.result_handles.wait_for_all_values()
    plt.figure(arg)
    freq_track_obj.time_domain_ramesy_full_sweep_analysis(job.result_handles, arg)
    print(freq_track_obj.fres)

# frequency domain ramsey
with program() as prog:
    freq_track_obj.qua_declarations()
    freq_track_obj.freq_domain_ramsey_full_sweep(
        reps,
        freq_track_obj.fres - 2 * freq_track_obj.f_ref,
        freq_track_obj.fres + 2 * freq_track_obj.f_ref,
        int(0.002e6),
        "Pe_fd",
        oscillation,
    )
    with stream_processing():
        freq_track_obj.state_estimation_st[0].buffer(reps, len(freq_track_obj.fvec)).map(FUNCTIONS.average()).save(
            "Pe_fd"
        )

qm = qmm.open_qm(config)
job = qm.execute(prog)
job.result_handles.wait_for_all_values()
plt.figure("Pe_fd")
freq_track_obj.freq_domain_ramsey_full_sweep_analysis(job.result_handles, "Pe_fd")

##################################################
# verify correction from 2point WRT ramsey in TD #
##################################################
# freq_track_obj.fres = freq_track_obj.fres - 0
# points = 50
#
# with program() as prog:
#
#     freq_track_obj.qua_declarations()
#     i = declare(int)
#     with for_(i, 0, i < points, i+1):
#         freq_track_obj.time_domain_ramesy_full_sweep(reps, freq_track_obj.f_ref, 4, 50000, 200, 'td')
#         freq_track_obj.two_points_ramsey()
#
#     with stream_processing():
#         freq_track_obj.state_estimation_st[0].buffer(reps, len(freq_track_obj.tau_vec)).map(FUNCTIONS.average()).save_all('td')
#         freq_track_obj.corr_st.save_all('2point')
#
# job = qm.execute(prog)
# job.result_handles.wait_for_all_values()
# Pe = job.result_handles.get('td').fetch_all()['value']
# td_shifts = []; two_point_sifts = []
# for i in range(points):
#     t = np.array(freq_track_obj.tau_vec)*4
#     out = qubit_frequency_tracking._fit_ramsey(freq_track_obj, t, Pe[i])  # in [ns]
#     td_shifts.append(out['f'] * 1e9 - freq_track_obj.f_ref)
#     print(f"td shift: {out['f'] * 1e9 - freq_track_obj.f_ref}")
#     two_point_sifts.append(job.result_handles.get('2point').fetch_all()['value'][i])
#     print(job.result_handles.get('2point').fetch_all()['value'][i])
#
# plt.figure('td ramsey corr VS two point ramsey corr'); plt.plot(td_shifts, two_point_sifts, '.')
# plt.figure('td ramsey & two-point ramsey vs time'); plt.plot(td_shifts); plt.plot(two_point_sifts)


###################
# goal experiment #
###################
with program() as prog:
    freq_track_obj.qua_declarations()
    i = declare(int)

    with for_(i, 0, i < 10000, i + 1):
        freq_track_obj.time_domain_ramesy_full_sweep(reps, freq_track_obj.f_ref, 4, 50000, 200, "Pe_td_ref", False)
        freq_track_obj.two_points_ramsey()
        freq_track_obj.time_domain_ramesy_full_sweep(reps, freq_track_obj.f_ref, 4, 50000, 200, "Pe_td_corr", True)

    with stream_processing():
        freq_track_obj.state_estimation_st[0].buffer(reps, len(freq_track_obj.tau_vec)).map(FUNCTIONS.average()).save(
            "Pe_td_ref"
        )
        freq_track_obj.state_estimation_st[1].buffer(reps, len(freq_track_obj.tau_vec)).map(FUNCTIONS.average()).save(
            "Pe_td_corr"
        )

job = qm.execute(prog)
td_ref_handle = job.result_handles.get("Pe_td_ref")
td_corr_handle = job.result_handles.get("Pe_td_corr")

td_ref_handle.wait_for_values(1)
td_corr_handle.wait_for_values(1)

t0 = time.time()
t0_local_time = time.localtime()
t_ = t0
hours = 2
cond = (t_ - t0) / 3600 < hours
fig, (ax1, ax2) = plt.subplots(1, 2)
Pe_td_ref = []
Pe_td_corr = []
t = []

while cond:
    Pe_td_ref_ = td_ref_handle.fetch_all()
    Pe_td_corr_ = td_corr_handle.fetch_all()
    t_ = time.time()
    t.append((t_ - t0) / 3600)
    Pe_td_ref.append(Pe_td_ref_)
    Pe_td_corr.append(Pe_td_corr_)
    ax1.pcolormesh(freq_track_obj.tau_vec, t, Pe_td_ref)
    ax1.title.set_text("TD Ramsey feedback off")
    ax1.set_xlabel("tau [ns]")
    ax1.set_ylabel("time [hours]")
    ax2.pcolormesh(freq_track_obj.tau_vec, t, Pe_td_corr)
    ax2.title.set_text("TD Ramsey feedback on")
    ax2.set_xlabel("tau [ns]")
    ax2.set_ylabel("time [hours]")
    plt.pause(10)

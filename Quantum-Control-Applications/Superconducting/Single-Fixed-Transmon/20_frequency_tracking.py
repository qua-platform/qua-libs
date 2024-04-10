from qm.qua import *
from qm import QuantumMachinesManager
from macros import qubit_frequency_tracking
from configuration import *
import matplotlib.pyplot as plt
import time
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close


######################################
#  Open Communication with the QOP  #
######################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

# Open quantum machine
qm = qmm.open_qm(config)

# Initialize object
freq_track_obj = qubit_frequency_tracking("qubit", "resonator", qubit_IF, ge_threshold, frame_rotation_flag=False)

################################
#  Step 1: Time domain Ramsey  #
################################
# This first step consists in perform a time Ramsey experiment to estimate the qubit frequency, phase of the oscillation and T2*.
n_avg = 20
tau_vec = np.arange(4, 50_000, 50)
print(f"Initial frequency: {freq_track_obj.f_res:.0f} Hz")

# Repeat the measurement twice, without and with correction of the frequency
for arg in ["Pe_initial", "Pe_corrected"]:
    # The QUA program
    with program() as prog:
        freq_track_obj.initialization()
        freq_track_obj.time_domain_ramsey_full_sweep(n_avg, f_det=int(0.06e6), tau_vec=tau_vec)

        with stream_processing():
            freq_track_obj.state_estimation_st[0].buffer(len(tau_vec)).average().save(arg)
    # Execute the program
    job = qm.execute(prog)
    # Wait until processing is done before fetching results
    job.result_handles.wait_for_all_values()
    # Plot raw data + fit
    plt.figure(arg)
    freq_track_obj.time_domain_ramsey_full_sweep_analysis(job.result_handles, stream_name=arg)
    # Prepare to apply a correction on the next iteration
    print(f"Correct frequency: {freq_track_obj.f_res:.0f} Hz")

#####################################
#  Step 2: Frequency domain Ramsey  #
#####################################
# This second step consist in performing a Ramsey experiment in the frequency domain (by sweeping the frequency of the
# pi/2 pulses) in order to measure the amplitude of the fringes and the points around the central fringe to perform the
# frequency lock in the next step.
n_avg = 20
f_min = freq_track_obj.f_res - 2 * freq_track_obj.f_det
f_max = freq_track_obj.f_res + 2 * freq_track_obj.f_det
d_f = 2 * u.kHz
f_vec = np.arange(f_min, f_max, d_f)
oscillation = 1

# The QUA program
with program() as prog:
    freq_track_obj.initialization()
    freq_track_obj.freq_domain_ramsey_full_sweep(n_avg, f_vec, oscillation)

    with stream_processing():
        freq_track_obj.state_estimation_st[0].buffer(len(f_vec)).average().save("Pe_fd")

# Execute the program
job = qm.execute(prog)
# Wait until processing is done before fetching results
job.result_handles.wait_for_all_values()
# Plot raw data + fit
plt.figure("Pe_fd")
freq_track_obj.freq_domain_ramsey_full_sweep_analysis(job.result_handles, "Pe_fd")

#################################
#  Step 3: Real-time correction #
#################################
# This third step consist in interleaving a two-point Ramsey measurement (Standard Ramsey measurement with the frequency
# detuned to be on both sides of the central fringe in frequency domain). Then depending on the measured population,
# one can estimate the frequency shift and update the QUA program accordingly to stay on resonance.
# In this example the two-point Ramsey is performed between two standard time Ramsey experiment in order to evaluate the
# quality of the tracking.
n_avg = 20
# Total duration of the experiment in minutes
minutes = 2
# Time between two successive run in seconds
time_between_two_runs = 10
# Time vector for the time domain Ramsey measurement
tau_vec = np.arange(4, 50_000, 200)
with program() as prog:
    i = declare(int)
    i_st = declare_stream()
    # with for_(i, 0, i < 100, i + 1):
    with infinite_loop_():
        freq_track_obj.initialization()
        freq_track_obj.time_domain_ramsey_full_sweep(n_avg, freq_track_obj.f_det, tau_vec, False)
        freq_track_obj.two_points_ramsey(n_avg_power_of_2=1)
        freq_track_obj.time_domain_ramsey_full_sweep(n_avg, freq_track_obj.f_det, tau_vec, True)
        assign(i, i + 1)
        save(i, i_st)
        pause()
    with stream_processing():
        freq_track_obj.state_estimation_st[0].buffer(len(tau_vec)).buffer(n_avg).map(FUNCTIONS.average()).save(
            "Pe_td_ref"
        )
        freq_track_obj.state_estimation_st[1].buffer(len(tau_vec)).buffer(n_avg).map(FUNCTIONS.average()).save(
            "Pe_td_corr"
        )
        i_st.save("iteration")
        freq_track_obj.f_res_corr_st.save_all("f_res_corr")
        freq_track_obj.corr_st.save_all("corr")

# Execute the program
job = qm.execute(prog)
# Handle results
results = fetching_tool(job, ["Pe_td_ref", "Pe_td_corr", "iteration", "f_res_corr", "corr"], mode="live")

# Starting time
t0 = time.time()
t_ = t0
cond = (t_ - t0) / 60 < minutes

# Initialize results lists
Pe_td_ref = []
Pe_td_corr = []
t = []
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)
while results.is_processing():
    if cond:
        # Fetch results
        Pe_td_ref_, Pe_td_corr_, iteration, f_res_corr, corr = results.fetch_all()
        # Progress bar
        progress_counter(iteration, int(minutes / (time_between_two_runs / 60)), start_time=t0)
        # Get current time
        t_ = time.time()
        # Update while loop condition
        cond = (t_ - t0) / 60 < minutes
        # Update time vector and results
        t.append((t_ - t0) / 60)
        Pe_td_ref.append(Pe_td_ref_)
        Pe_td_corr.append(Pe_td_corr_)
        time.sleep(time_between_two_runs)
        job.resume()
    else:
        job.halt()
    # Plot results
    plt.subplot(121)
    plt.pcolormesh(freq_track_obj.tau_vec, t, Pe_td_ref)
    plt.title("TD Ramsey feedback off")
    plt.xlabel("tau [ns]")
    plt.ylabel("time [minutes]")
    plt.subplot(122)
    plt.pcolormesh(freq_track_obj.tau_vec, t, Pe_td_corr)
    plt.title("TD Ramsey feedback on")
    plt.xlabel("tau [ns]")
    plt.ylabel("time [minutes]")
    plt.tight_layout()
    plt.pause(0.01)

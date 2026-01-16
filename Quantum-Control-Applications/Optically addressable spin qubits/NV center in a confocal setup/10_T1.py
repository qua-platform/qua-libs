"""
       T1 MEASUREMENT
The program consists in measuring the photon counts (in |0> and |1> successively) received by the SPCM across
varying wait times either after initialization (start from |0>), or after a pi pulse (start from |1>).

The data is then post-processed to determine the thermal relaxation time T1.

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Having set the pi pulse amplitude and duration in the configuration

Next steps before going to the next node:
    -
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
t_vec = np.arange(4, 250, 10)  # The wait time vector in clock cycles (4ns)
n_avg = 1_000_000  # The number averaging iterations
start_from_one = False

# Determine reference readout during single laser pulse
reference_wait = initialization_len_1 // 4 - 2 * meas_len_1 // 4 - 25  # in clock cycles
reference_readout = reference_wait >= 4

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_vec": t_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as T1:
    counts = declare(int)  # saves number of photon counts
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    counts_1_ref_st = declare_stream()  # stream for counts
    counts_2_ref_st = declare_stream()  # stream for counts
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    # T1 sequence
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            # Measure in |0>
            if start_from_one:  # Choose to start either from |0> or |1>
                play("x180" * amp(1), "NV")
            wait(t, "NV")  # variable delay before measurement
            # To measure in |0> keeping the sequence time constant
            play("x180" * amp(0), "NV")
            align()  # Play the laser pulse after the mw sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_1_st)  # save counts
            # Measure reference photon counts at end of laser pulse
            if reference_readout:
                wait(reference_wait, "SPCM1")
                measure("readout", "SPCM1", time_tagging.analog(times, meas_len_1, counts))
            else:
                assign(counts, 1)
            save(counts, counts_1_ref_st)

            wait(wait_between_runs * u.ns, "AOM1")

            align()
            # Measure in |1>
            if start_from_one:  # Choose to start either from |0> or |1>
                play("x180" * amp(1), "NV")
            wait(t, "NV")  # variable delay in spin Echo
            # To measure in |1>
            play("x180" * amp(1), "NV")
            align()  # Play the laser pulse after the mw sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_2_st)  # save counts
            # Measure reference photon counts at end of laser pulse
            if reference_readout:
                wait(reference_wait, "SPCM1")
                measure("readout", "SPCM1", time_tagging.analog(times, meas_len_1, counts))
            else:
                assign(counts, 1)
            save(counts, counts_2_ref_st)

            wait(wait_between_runs * u.ns, "AOM1")

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_1_ref_st.buffer(len(t_vec)).average().save("counts1_ref")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        counts_2_ref_st.buffer(len(t_vec)).average().save("counts2_ref")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, T1, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open the quantum machine
    qm = qmm.open_qm(config, close_other_machines=True)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(T1)
    # Get results from QUA program
    results = fetching_tool(
        job, data_list=["counts1", "counts1_ref", "counts2", "counts2_ref", "iteration"], mode="live"
    )
    # Live plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts1_ref, counts2, counts2_ref, counts_dark, iteration = results.fetch_all()
        # Compute normalized signals
        norm1 = counts1 / counts1_ref
        norm2 = counts2 / counts2_ref
        diff = norm1 - norm2
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        # Plot data
        ax1.cla()
        ax1.plot(4 * t_vec, norm1, label="counts in |0>")
        ax1.plot(4 * t_vec, norm2, label="counts in |1>")
        ax1.set_ylabel("Norm. Signal")
        ax1.set_title("Ramsey")
        ax1.legend()

        ax2.cla()
        ax2.plot(4 * t_vec, diff, color="black", label="Difference")
        ax2.set_xlabel("Wait time [ns]")
        ax2.set_ylabel("ΔSignal")
        ax2.legend()
        plt.pause(0.1)

    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts1_data": counts1})
    save_data_dict.update({"counts1_ref_data": counts1_ref})
    save_data_dict.update({"normalized1_data": norm1})
    save_data_dict.update({"counts2_data": counts2})
    save_data_dict.update({"counts2_ref_data": counts2_ref})
    save_data_dict.update({"normalized2_data": norm2})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

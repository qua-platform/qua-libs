"""
        POWER RABI
The program consists in playing a mw pulse and measure the photon counts received by the SPCM
across varying mw pulse amplitudes.
The sequence is repeated without playing the mw pulses to measure the dark counts on the SPCM.

The data is then post-processed to determine the pi pulse amplitude for the specified duration.

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Set the desired pi pulse duration, labeled as "mw_len_NV", in the configuration

Next steps before going to the next node:
    - Update the pi pulse amplitude, labeled as "mw_amp_NV", in the configuration.
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
a_vec = np.arange(0.1, 1, 0.02)  # The amplitude pre-factor vector
n_avg = 1_000_000  # number of iterations

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "a_vec": a_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as power_rabi:
    counts = declare(int)  # variable for number of counts
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    a = declare(fixed)  # variable to sweep over the amplitude
    n = declare(int)  # variable to for_loop
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    # Power Rabi sweep
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, a_vec)):
            # Play the Rabi pulse with varying amplitude
            play("x180" * amp(a), "NV")  # 'a' is a pre-factor to the amplitude defined in the config ("mw_amp_NV")
            align()  # Play the laser pulse after the mw pulse
            play("laser_ON", "AOM1")
            # Measure and detect the photons on SPCM1
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts

            # Wait and align all elements before measuring the dark events
            wait(wait_between_runs * u.ns)
            align()

            # Play the Rabi pulse with zero amplitude
            play("x180" * amp(0), "NV")
            align()  # Play the laser pulse after the mw pulse
            play("laser_ON", "AOM1")
            # Measure and detect the dark counts on SPCM1
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_dark_st)  # save dark counts
            wait(wait_between_runs * u.ns)  # wait in between iterations

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_st.buffer(len(a_vec)).average().save("counts")
        counts_dark_st.buffer(len(a_vec)).average().save("counts_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, power_rabi, simulation_config)
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
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(power_rabi)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "counts_dark", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts, counts_dark, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(a_vec * x180_amp_NV, counts / 1000 / (meas_len_1 * 1e-9), label="photon counts")
        plt.plot(a_vec * x180_amp_NV, counts_dark / 1000 / (meas_len_1 * 1e-9), label="dark_counts")
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Power Rabi")
        plt.legend()
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts_data": counts})
    save_data_dict.update({"counts_dark_data": counts_dark})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

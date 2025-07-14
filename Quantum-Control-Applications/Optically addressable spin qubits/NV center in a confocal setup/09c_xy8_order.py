"""
       XY8 MEASUREMENT (order sweep)
The program consists in playing two XY8 sequences successively (first ending with x90 and then with -x90)
and measure the photon counts received by the SPCM across varying the order n of the XY8-n measurement.
Note that the pulse spacing (tau) is the interpulse spacing, i.e. the time between consecutive pi-pulses and
does not take into account the finite length of the pi-pulses.

The data is post-processed to determine the coherence time T2 associated with the XY8 order sweep measurement.

The sequence is defined in the following way:
pix/2 - [t - pix - 2t - piy - 2t - pix - 2t - piy - 2t - piy - 2t - pix - 2t - piy - 2t - pix - t] ^ xy8_order - pix/2

There is a possible timing gap due to QUA's pulse processing between the initial pi/2-pulse and the XY8 block.
To evaluate the length of the gap the `with strict_timing_()` command can be used and the gap duration can
be extracted from the error message.
To compensate the gap, the gap duration is added to the time between the XY8 block and the last pi/2-pulse.

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
from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.results.data_handler import DataHandler

import matplotlib.pyplot as plt


##################
#   Parameters   #
##################
tau = 2 * 50  # The time between pi-pulses in clock cycles (4ns)
order_vec = np.arange(1, 21, 1, dtype=int)  # order vector for varying the order n of the XY8-n measurement
n_avg = 1_000_000
t_gap = 10  # possible timing gap in clock cycles that is to be compensated
tau_half = tau // 2  # time between pi/2-pulse and XY8 block

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "order_vec": order_vec,
    "config": config,
}


##########################
#  XY8 sequence wrapper  #
##########################
def xy8_block():
    # A single XY8 block, ends at x frame.
    play("x180", "NV")  # 1 X
    wait(tau, "NV")

    play("y180", "NV")  # 2 Y
    wait(tau, "NV")

    play("x180", "NV")  # 3 X
    wait(tau, "NV")

    play("y180", "NV")  # 4 Y
    wait(tau, "NV")

    play("y180", "NV")  # 5 Y
    wait(tau, "NV")

    play("x180", "NV")  # 6 X
    wait(tau, "NV")

    play("y180", "NV")  # 7 Y
    wait(tau, "NV")

    play("x180", "NV")  # 8 X


###################
# The QUA program #
###################
with program() as xy8_order_sweep:
    xy8_order = declare(int)  # order of the XY8 measurement
    n = declare(int)  # for averages
    i = declare(int)  # for xy8 order

    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)  # QUA vector for storing the time-tags
    times2 = declare(int, size=100)  # QUA vector for storing the time-tags
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    # XY8-n sequence
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(xy8_order, order_vec)):
            play("x90", "NV")
            with for_(i, 0, i < xy8_order, i + 1):
                wait(tau_half, "NV")
                xy8_block()
                wait(tau_half, "NV")
            wait(t_gap, "NV")  # compensate for timing gap
            play("x90", "NV")
            align()  # Play the laser pulse after the XY8 sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times1, meas_len_1, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            align()
            play("x90", "NV")
            with for_(i, 0, i < xy8_order, i + 1):
                wait(tau_half, "NV")
                xy8_block()
                wait(tau_half, "NV")
            wait(t_gap, "NV")  # compensate for timing gap
            play("-x90", "NV")
            align()  # Play the laser pulse after the Echo sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times2, meas_len_1, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

        save(n, n_st)  # save number of iterations inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_1_st.buffer(len(order_vec)).average().save("counts1")
        counts_2_st.buffer(len(order_vec)).average().save("counts2")
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
    job = qmm.simulate(config, xy8_order_sweep, simulation_config)
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
    # execute QUA program
    job = qm.execute(xy8_order_sweep)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts1", "counts2", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts2, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(2 * order_vec, counts1 / 1000 / (meas_len_1 / u.s), label="x90_XY8_x90")
        plt.plot(2 * order_vec, counts2 / 1000 / (meas_len_1 / u.s), label="x90_XY8_-x90")
        plt.xlabel("XY8 order")
        plt.ylabel("Intensity [kcps]")
        plt.title("XY8 order sweep")
        plt.legend()
        plt.pause(0.1)

        plt.cla()
        contrast = (counts2 - counts1) / 1000 / (meas_len_1 / u.s)
        plt.plot(2 * order_vec, contrast, label="XY8 contrast")
        plt.xlabel("XY8 order")
        plt.ylabel("Contrast [kcps]")
        plt.title("XY8 order sweep")
        plt.legend()
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts1_data": counts1})
    save_data_dict.update({"counts2_data": counts2})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

"""
       XY8 MEASUREMENT (tau sweep)
The program consists in playing two XY8 sequences successively (first ending with x90 and then with -x90)
and measure the photon counts received by the SPCM across varying idle times between pi-pulses.
Note that the pulse spacing tau is the interpulse spacing, i.e. the time between consecutive pi-pulses and
does not take into account the finite length of the pi-pulses.

The data is post-processed to determine the coherence time T2 associated with the XY8 tau sweep measurement.

The sequence is defined in the following way:
pix/2 - [t - pix - 2t - piy - 2t - pix - 2t - piy - 2t - piy - 2t - pix - 2t - piy - 2t - pix - t] ^ xy8_order - pix/2

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
tau_vec = 2 * np.arange(4, 500, 20)  # The time vector for varying the time between pulses in clock cycles (4ns)
xy8_order = 4  # order n of the XY8-n measurement
n_avg = 1_000_000

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_vec": tau_vec,
    "config": config,
}

###########################
#  XY8 sequence wrappers  #
###########################
def xy8_n(order):
    """Performs the full xy8_n sequence.
    The first block is outside loop to avoid delays caused from either the
    loop or from two consecutive wait commands.
    Assumes it starts frame at x, if not, need to reset_frame before"""
    wait(tau_half, "NV")
    xy8_block()
    for _ in range(order - 1):
        wait(tau, "NV")
        xy8_block()
    wait(tau_half, "NV")

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
with program() as xy8_tau:
    tau = declare(int)  # interpulse spacing
    tau_half = declare(int)  # half interpulse spacing
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
        with for_(*from_array(tau, tau_vec)):
            assign(tau_half, tau / 2)
            with strict_timing_():
                # First XY8 sequence with x90 - XY8 block - x90
                play("x90", "NV")
                xy8_n(xy8_order)
                play("x90", "NV")
            align()  # Play the laser pulse after the XY8 sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times1, meas_len_1, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            align()
            # Second XY8 sequence with x90 - XY8 block - -x90
            with strict_timing_():
                # First XY8 sequence with x90 - XY8 block - x90
                play("x90", "NV")
                xy8_n(xy8_order)
                play("-x90", "NV")
            align()  # Play the laser pulse after the Echo sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times2, meas_len_1, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")

            wait(wait_between_runs * u.ns, "AOM1")

        save(n, n_st)  # save number of iterations inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_1_st.buffer(len(tau_vec)).average().save("counts1")
        counts_2_st.buffer(len(tau_vec)).average().save("counts2")
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
    job = qmm.simulate(config, xy8_tau, simulation_config)
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
    job = qm.execute(xy8_tau)
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
        plt.plot(2 * tau_vec, counts1 / 1000 / (meas_len_1 / u.s), label="x90_XY8_x90")
        plt.plot(2 * tau_vec, counts2 / 1000 / (meas_len_1 / u.s), label="x90_XY8_-x90")
        plt.xlabel("Interpulse spacing [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.title("XY8-{} tau sweep".format(xy8_order))
        plt.legend()
        plt.pause(0.1)

        plt.cla()
        contrast = (counts2 - counts1) / 1000 / (meas_len_1 / u.s)
        plt.plot(2 * tau_vec, contrast, label="XY8 contrast")
        plt.xlabel("Interpulse spacing [ns]")
        plt.ylabel("Contrast [kcps]")
        plt.title("XY8-{} tau sweep".format(xy8_order))
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

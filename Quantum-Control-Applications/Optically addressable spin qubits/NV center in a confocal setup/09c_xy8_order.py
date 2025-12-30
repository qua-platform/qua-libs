"""
       XY8 MEASUREMENT (order sweep)
The program consists in playing two XY8 sequences successively (first ending with x90 and then with -x90)
and measure the photon counts received by the SPCM across varying the order n of the XY8-n measurement.
The time `tau` is the time between pi-pulse centers. From this the pulse spacing is calculated by subtracting
the duration of the pi-pulse. The same is done for tau_half. It is assumed that the pi/2-pulse has the same length.

The data is post-processed to determine the coherence time T2 associated with the XY8 order sweep measurement.

The sequence is defined in the following way:
x90 - [t - x180 - 2t - y180 - 2t - x180 - 2t - y180 - 2t - y180 - 2t - x180 - 2t - y180 - 2t - x180 - t] ^ xy8_order - x90

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

import logging
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
qm_log = logging.getLogger("qm")


##################
#   Parameters   #
##################
# The time between pi-pulse centers in clock cycles (4ns)
# Must be a multiple of 2 clock cycles to ensure that tau_half is a multiple of a single clock cycle
tau = 2 * 50
order_vec = np.arange(1, 21, 1, dtype=int)  # order vector for varying the order n of the XY8-n measurement
n_avg = 1_000_000

tau_spacing = tau - x180_len_NV  # interpulse spacing, i.e. from end of pulse to beginning of next pulse
tau_half_spacing = tau // 2 - x90_len_NV  # interpulse spacing for tau_half
if x180_len_NV != x90_len_NV:
    qm_log.warning(
        f"pi-pulse and pi/2-pulse do not have the same length ({x180_len_NV}, {x90_len_NV}). "
        f"For physical correctness they should be the same."
    )

if tau_half_spacing < 4:
    raise ValueError(f"Interpulse spacing must be >= 4. Got ({tau_half_spacing}) for tau half.")

# Determine reference readout during single laser pulse
reference_wait = initialization_len_1 // 4 - 2 * meas_len_1 // 4 - 25  # in clock cycles
reference_readout = reference_wait >= 4

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "tau": tau,
    "order_vec": order_vec,
    "config": config,
}


##########################
#  XY8 sequence wrapper  #
##########################
def xy8_n(tau, order):
    """Performs the full xy8_n sequence.
    The first block is outside the loop to avoid delays caused from either the
    loop or from two consecutive wait commands."""
    xy8_block(tau)
    with for_(i, 1, i <= order - 1, i + 1):
        wait(tau, "NV")
        xy8_block(tau)


def xy8_block(tau):  # A single XY8 block
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

    counts = declare(int)  # saves number of photon counts
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    counts_1_ref_st = declare_stream()  # stream for counts
    counts_2_ref_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    # Spin initialization
    play("laser_ON", "AOM1")
    wait(wait_for_initialization * u.ns, "AOM1")

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(xy8_order, order_vec)):
            # Strict_timing validates that the sequence will be played without gaps.
            # If gaps are detected, an error will be raised
            with strict_timing_():
                # First XY8 sequence with x90 - XY8-order block - x90
                play("x90", "NV")
                wait(tau_half_spacing, "NV")
                xy8_n(tau_spacing, xy8_order)
                wait(tau_half_spacing, "NV")
                play("x90", "NV")
            align()  # Play the laser pulse after the XY8 sequence
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
            with strict_timing_():
                # Second XY8 sequence with x90 - XY8-order block - -x90
                play("x90", "NV")
                wait(tau_half_spacing, "NV")
                xy8_n(tau_spacing, xy8_order)
                wait(tau_half_spacing, "NV")
                play("-x90", "NV")
            align()  # Play the laser pulse after the Echo sequence
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

        save(n, n_st)  # save number of iterations inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_1_st.buffer(len(order_vec)).average().save("counts1")
        counts_1_ref_st.buffer(len(order_vec)).average().save("counts1_ref")
        counts_2_st.buffer(len(order_vec)).average().save("counts2")
        counts_2_ref_st.buffer(len(order_vec)).average().save("counts2_ref")
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
    results = fetching_tool(
        job, data_list=["counts1", "counts1_ref", "counts2", "counts2_ref", "iteration"], mode="live"
    )
    # Live plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts1_ref, counts2, counts2_ref, iteration = results.fetch_all()
        # Compute normalized signals
        norm1 = counts1 / counts1_ref
        norm2 = counts2 / counts2_ref
        diff = norm1 - norm2
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        # Plot data
        ax1.cla()
        ax1.plot(order_vec, norm1, label="x90_XY8-n_x90")
        ax1.plot(order_vec, norm2, label="x90_XY8-n_-x90")
        ax1.set_ylabel("Norm. Signal")
        ax1.set_title("XY8 order sweep")
        ax1.legend()

        ax2.cla()
        ax2.plot(order_vec, diff, color="black", label="Difference")
        ax2.set_xlabel("XY8 order")
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
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

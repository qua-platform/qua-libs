"""
        CW Optically Detected Magnetic Resonance (ODMR)
The program consists in playing a mw pulse and the readout laser pulse simultaneously to extract
the photon counts received by the SPCM across varying intermediate frequencies.
The sequence is repeated without playing the mw pulses to measure the dark counts on the SPCM.

The data is then post-processed to determine the spin resonance frequency.
This frequency can be used to update the NV intermediate frequency in the configuration under "NV_IF_freq".

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Update the different delays in the configuration

Next steps before going to the next node:
    - Update the NV frequency, labeled as "NV_IF_freq", in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *


###################
# The QUA program #
###################

# Frequency vector
f_vec = np.arange(-30 * u.MHz, 70 * u.MHz, 2 * u.MHz)
n_avg = 1_000_000  # number of averages
readout_len = long_meas_len_1  # Readout duration for this experiment

with program() as cw_odmr:
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, f_vec)):
            # Update the frequency of the digital oscillator linked to the element "NV"
            update_frequency("NV", f)
            # align all elements before starting the sequence
            align()
            # Play the mw pulse...
            play("cw" * amp(1), "NV", duration=readout_len * u.ns)
            # ... and the laser pulse simultaneously (the laser pulse is delayed by 'laser_delay_1')
            play("laser_ON", "AOM1", duration=readout_len * u.ns)
            wait(1_000 * u.ns, "SPCM1")  # so readout don't catch the first part of spin reinitialization
            # Measure and detect the photons on SPCM1
            measure("long_readout", "SPCM1", None, time_tagging.analog(times, readout_len, counts))

            save(counts, counts_st)  # save counts on stream

            # Wait and align all elements before measuring the dark events
            wait(wait_between_runs * u.ns)
            align()  # align all elements
            # Play the mw pulse with zero amplitude...
            play("cw" * amp(0), "NV", duration=readout_len * u.ns)
            # ... and the laser pulse simultaneously (the laser pulse is delayed by 'laser_delay_1')
            play("laser_ON", "AOM1", duration=readout_len * u.ns)
            wait(1_000 * u.ns, "SPCM1")  # so readout don't catch the first part of spin reinitialization
            measure("long_readout", "SPCM1", None, time_tagging.analog(times, readout_len, counts))

            save(counts, counts_dark_st)  # save counts on stream

            wait(wait_between_runs * u.ns)

            save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        counts_st.buffer(len(f_vec)).average().save("counts")
        counts_dark_st.buffer(len(f_vec)).average().save("counts_dark")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cw_odmr, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cw_odmr)
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
        plt.plot((NV_LO_freq * 0 + f_vec) / u.MHz, counts / 1000 / (readout_len * 1e-9), label="photon counts")
        plt.plot((NV_LO_freq * 0 + f_vec) / u.MHz, counts_dark / 1000 / (readout_len * 1e-9), label="dark counts")
        plt.xlabel("MW frequency [MHz]")
        plt.ylabel("Intensity [kcps]")
        plt.title("ODMR")
        plt.legend()
        plt.pause(0.1)

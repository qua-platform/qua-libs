"""
        CHARGE STABILITY MAP - fast axis: OPX (AC) & slow axis: external source (DC)
The goal of the script is to acquire the charge stability map.
Here the charge stability diagram is acquired by sweeping the fast axis with the OPX connected to the AC part of the
bias-tee, while the slow axis is handled by an external source connected to the DC part of the bias-tee.

This is done by pausing the QUA program, updating the voltages in Python using the instrument API and resuming the QUA program.
The OPX is simply measuring, either via dc current sensing or RF reflectometry, the charge occupation of the dot.

A single-point averaging is performed and the data is extracted while the program is running to display the results line-by-line.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the external DC source using its driver.
    - Connect one plunger gate (DC line of the bias-tee) to the external dc source and the other plunger gate (AC line of the bias-tee) to tne OPX.

Before proceeding to the next node:
    - Identify the different charge occupation regions
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro, get_filtered_voltage

###################
# The QUA program #
###################
n_avg = 100
n_points_slow = 10
n_points_fast = 101

# Voltages in Volt
voltage_values_slow = np.linspace(-1.5, 1.5, n_points_slow)
# Because of the bias-tee, it is important that the voltages swept along the fast axis are centered around 0.
# Also, since the OPX dynamic range is [-0.5, 0.5)V, one may need to add a voltage offset on the DC part of the bias-tee.
voltage_values_fast = np.linspace(-0.2, 0.2, n_points_fast)
# TODO: set DC offset on the external source for the fast gate
# One can check the expected voltage levels after the bias-tee using the following function:
_, _ = get_filtered_voltage(voltage_values_fast, step_duration=1e-6, bias_tee_cut_off_frequency=1e3, plot=True)

with program() as charge_stability_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    Vfast = declare(fixed)  # QUA fixed to increment the voltage offset handled by one OPX channel
    n_st = declare_stream()  # Stream for the iteration number (progress bar)

    with for_(i, 0, i < n_points_slow + 1, i + 1):
        # Pause the OPX to update the external DC voltages in Python
        pause()
        # Wait for the voltages to settle (depends on the voltage source bandwidth)
        wait(1 * u.ms)
        with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
            with for_(*from_array(Vfast, voltage_values_fast)):
                # Update the dc offset of the specified element
                set_dc_offset("P1", "single", Vfast)
                # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
                # frequency and the integrated quadratures are stored in "I" and "Q"
                I, Q, I_st, Q_st = RF_reflectometry_macro()
                # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
                # is stored in "dc_signal"
                dc_signal, dc_signal_st = DC_current_sensing_macro()
                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
                # process them which can cause the OPX to crash.
                wait(1_000 * u.ns)  # in ns
        # Save the LO iteration to get the progress bar
        save(i, n_st)

    # Stream processing section used to process the data before saving it
    with stream_processing():
        n_st.save("iteration")
        # Perform a single point averaging and cast the data into a 1D array. "save_all" is used here to store all
        # received 1D arrays, and eventually form a 2D array, which enables line-by-line live plotting
        # RF reflectometry
        I_st.buffer(n_points_fast).buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
        Q_st.buffer(n_points_fast).buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")
        # DC current sensing
        dc_signal_st.buffer(n_points_fast).buffer(n_avg).map(FUNCTIONS.average()).save_all("dc_signal")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, charge_stability_prog, simulation_config)
    plt.figure()
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(charge_stability_prog)
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    for i in range(n_points_slow):  # Loop over y-voltages
        # Set voltage
        # TODO: update slow axis voltage with external dc source
        # qdac.write(f"sour{qdac_channel_slow}:volt {voltage_values_slow[i]}")
        # Resume the QUA program (escape the 'pause' statement)
        job.resume()
        # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
        wait_until_job_is_paused(job)
        # Get results from QUA program and initialize live plotting
        results = fetching_tool(job, data_list=["I", "Q", "dc_signal", "iteration"], mode="live")

        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, DC_signal, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, reflectometry_readout_length, single_demod=True)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        DC_signal = u.demod2volts(DC_signal, readout_len, single_demod=True)
        # Progress bar
        progress_counter(iteration, n_points_slow, start_time=results.start_time)
        # Plot data
        plt.subplot(121)
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.pcolor(voltage_values_fast, voltage_values_slow, R)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(voltage_values_fast, voltage_values_slow, phase)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.tight_layout()
        plt.pause(0.1)

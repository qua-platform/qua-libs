"""
        Pauli Spin Blockade search
The goal of the script is to find the PSB region according to the protocol described in Nano Letters 2020 20 (2), 947-952.
To do so, the charge stability map is acquired by scanning the voltages provided by an external DC source,
to the DC part of the bias-tees connected to the plunger gates, while 2 OPX channels are stepping the voltages on the fast
lines of the bias-tees to navigate through the triangle in voltage space (empty - random initialization - measurement).

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage in average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time. A function has been written for this.

In the current implementation, the OPX is also measuring (either with DC current sensing or RF-reflectometry) during the
readout window (last segment of the triangle).
A single-point averaging is performed and the data is extracted while the program is running to display the results line-by-line.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the external DC source using its driver.
    - Connect the two plunger gates (DC line of the bias-tee) to the external dc source.
    - Connect the OPX to the fast line of the plunger gates for playing the triangle pulse sequence.

Before proceeding to the next node:
    - Identify the PSB region and update the config.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
from macros import RF_reflectometry_macro, DC_current_sensing_macro
import matplotlib.pyplot as plt

###################
# The QUA program #
###################
n_avg = 100  # Number of averages
n_points_slow = 101  # Number of points for the slow axis
n_points_fast = 100  # Number of points for the fast axis
Coulomb_amp = 0.0  # amplitude of the Coulomb pulse
# How many Coulomb pulse periods to last the whole program
N = (int((readout_len + 1_000) / (2 * step_length)) + 1) * n_points_fast * n_points_slow * n_avg

# Points in the charge stability map [V1, V2]
level_empty = [-0.2, 0.0]
duration_empty = 5000

seq = VoltageGateSequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("empty", level_empty, duration_empty)
seq.add_points("initialization", level_init, duration_init)
seq.add_points("readout", level_readout, duration_readout)

# Voltages in Volt
voltage_values_slow = np.linspace(-1.5, 1.5, n_points_slow)
voltage_values_fast = np.linspace(-1.5, 1.5, n_points_fast)

with program() as PSB_search_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    counter = declare(int)  # QUA integer used as an index for the Coulomb pulse
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    j = declare(int)  # QUA integer used as an index to loop over the voltage points
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    I = declare(fixed)
    Q = declare(fixed)
    dc_signal = declare(fixed)

    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("tank_circuit", I, Q)
    assign_variables_to_element("TIA", dc_signal)

    with for_(i, 0, i < n_points_slow + 1, i + 1):
        with for_(j, 0, j < n_points_fast, j + 1):
            # Pause the OPX to update the external DC voltages in Python
            pause()
            # Wait for the voltages to settle (depends on the voltage source bandwidth)
            wait(1 * u.ms)

            # Play the triangle
            seq.add_step(voltage_point_name="empty")
            seq.add_step(voltage_point_name="initialization")
            seq.add_step(voltage_point_name="readout")
            seq.add_compensation_pulse(duration=duration_compensation_pulse)
            # Measure the dot right after the qubit manipulation
            wait((duration_init + duration_empty) * u.ns, "tank_circuit", "TIA")
            # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
            # frequency and the integrated quadratures are stored in "I" and "Q"
            I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q)
            # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
            # is stored in "dc_signal"
            dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal)
            # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
            # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
            # process them which can cause the OPX to crash.
            wait(1_000 * u.ns)  # in ns
            # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
            ramp_to_zero("P1_sticky")
            ramp_to_zero("P2_sticky")
        # Save the LO iteration to get the progress bar
        save(i, n_st)

    # Stream processing section used to process the data before saving it
    with stream_processing():
        n_st.save("iteration")
        # Perform a single point averaging and cast the data into a 1D array. "save_all" is used here to store all
        # received 1D arrays, and eventually form a 2D array, which enables line-by-line live plotting
        # RF reflectometry
        I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("I")
        Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("Q")
        # DC current sensing
        dc_signal_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("dc_signal")

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
    simulation_config = SimulationConfig(duration=100_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, PSB_search_prog, simulation_config)
    plt.figure()
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(PSB_search_prog)
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    for i in range(n_points_slow):  # Loop over y-voltages
        # Set voltage
        # TODO: update slow axis voltage with external dc source
        # qdac.write(f"sour{qdac_channel_slow}:volt {voltage_values_slow[i]}")
        for j in range(n_points_fast):  # Loop over x-voltages
            # Set voltage
            # TODO: update fast axis voltage with external dc source
            # qdac.write(f"sour{qdac_channel_fast}:volt {voltage_values_fast[j]}")
            # Resume the QUA program (escape the 'pause' statement)
            job.resume()
            # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
            wait_until_job_is_paused(job)
        if i == 0:
            # Get results from QUA program and initialize live plotting
            results = fetching_tool(job, data_list=["I", "Q", "dc_signal", "iteration"], mode="live")

        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, DC_signal, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I[: iteration + 1] + 1j * Q[: iteration + 1], reflectometry_readout_length, single_demod=True)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        DC_signal = u.demod2volts(DC_signal, readout_len, single_demod=True)
        # Progress bar
        progress_counter(iteration, n_points_slow)
        # Plot data
        plt.subplot(121)
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.pcolor(voltage_values_fast, voltage_values_slow[: iteration + 1], R)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(voltage_values_fast, voltage_values_slow[: iteration + 1], phase)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.tight_layout()
        plt.pause(0.1)

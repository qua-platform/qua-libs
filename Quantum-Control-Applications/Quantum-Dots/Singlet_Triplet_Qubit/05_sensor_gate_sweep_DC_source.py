"""
        CHARGE SENSOR GATE SWEEP with an external DC source
Here the voltage biasing the sensor gate is provided and being swept by an external DC source connected to the DC line
of the bias-tee.
The OPX is simply measuring, either via dc current sensing or RF reflectometry, the response of the sensor dot.

A single point averaging is performed (averaging on the most inner loop) and the data is extracted while the program is
running.

Prerequisites:
    - Connect one the DC line of the bias-tee connected to the sensor dot to one OPX channel.
    - Setting the parameters of the external DC source using its driver if needed.

Before proceeding to the next node:
    - Update the config with the optimal sensing point.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # Number of averaging loops
n_points = 101
offsets = np.linspace(-0.2, 0.2, n_points)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "offsets": offsets,
    "config": config,
}

###################
# The QUA program #
###################
with program() as charge_sensor_sweep:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    j = declare(int)  # QUA integer used as an index to loop over the voltage points
    n_st = declare_stream()  # Stream for the iteration number (progress bar)

    with for_(i, 0, i < n_points + 1, i + 1):
        # Pause the OPX to update the external DC voltages in Python
        pause()
        # Wait for the voltages to settle (depends on the voltage source bandwidth)
        wait(1 * u.ms)

        with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
            # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
            # frequency and the integrated quadratures are stored in "I" and "Q"
            I, Q, I_st, Q_st = RF_reflectometry_macro()
            # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
            # is stored in "dc_signal"
            dc_signal, dc_signal_st = DC_current_sensing_macro()
            # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
            # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
            # process them which can cause the OPX to crash.
            wait(1_000 * u.ns, "tank_circuit")
        # Save the LO iteration to get the progress bar
        save(i, n_st)

    with stream_processing():
        n_st.save("iteration")
        # Perform a single point averaging and cast the data into a 1D array. "save_all" is used here to store all
        # received data, and eventually form a 1D array, which enables live plotting
        # RF reflectometry
        I_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
        Q_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")
        # DC current sensing
        dc_signal_st.buffer(n_avg).map(FUNCTIONS.average()).save_all("dc_signal")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, charge_sensor_sweep, simulation_config)
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
    job = qm.execute(charge_sensor_sweep)
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    for i in range(n_points):  # Loop over y-voltages
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
    S = u.demod2volts(I + 1j * Q, reflectometry_readout_length, single_demod=True)
    R = np.abs(S)  # Amplitude
    phase = np.angle(S)  # Phase
    DC_signal = u.demod2volts(DC_signal, readout_len, single_demod=True)
    # Progress bar
    progress_counter(iteration, n_points)
    # Plot results
    plt.suptitle("Charge sensor gate sweep")
    plt.subplot(211)
    plt.cla()
    plt.plot(offsets[: iteration + 1], R)
    plt.xlabel("Sensor gate voltage [V]")
    plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
    plt.subplot(212)
    plt.cla()
    plt.plot(offsets[: iteration + 1], phase)
    plt.xlabel("Sensor gate voltage [V]")
    plt.ylabel("Phase [rad]")
    plt.tight_layout()
    plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"DC_signal_data": DC_signal})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

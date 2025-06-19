"""
        CHARGE STABILITY MAP - fast and slow axes: external source (DC)
The goal of the script is to acquire the charge stability map.
Here the charge stability diagram is acquired by sweeping the voltages using an external DC source (QDAC or else).
This is done by pausing the QUA program, updating the voltages in Python using the instrument API and resuming the QUA program.

The OPX is simply measuring, either via dc current sensing or RF reflectometry, the charge occupation of the dot.
On top of the DC voltage sweeps, the OPX can output a continuous square wave (Coulomb pulse) through the AC line of the
bias-tee. This allows to check the coupling of the fast line to the sample and measure the lever arms between the DC and
AC lines.

A single-point averaging is performed and the data is extracted while the program is running to display the results line-by-line.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the external DC source using its driver.
    - Connect the two plunger gates (DC line of the bias-tee) to the external dc source.
    - (optional) Connect the OPX to the fast line of the plunger gates for playing the Coulomb pulse and calibrate the
      lever arm.

Before proceeding to the next node:
    - Identify the different charge occupation regions.
    - Update the config with the lever-arms.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100
n_points_slow = 10
n_points_fast = 11
Coulomb_amp = 0.1  # amplitude of the Coulomb pulse
# How many Coulomb pulse periods to last the whole program
N = (int((readout_len + 1_000 + 0 * 1 * u.ms) / (2 * step_length)) + 1) * n_avg

# Voltages in Volt
voltage_values_slow = np.linspace(-1.5, 1.5, n_points_slow)
voltage_values_fast = np.linspace(-1.5, 1.5, n_points_fast)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "Coulomb_amp": Coulomb_amp,
    "N_coulomb_pulses": N,
    "voltage_values_slow": voltage_values_slow,
    "voltage_values_fast": voltage_values_fast,
    "config": config,
}

###################
# The QUA program #
###################
with program() as charge_stability_prog:
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

            # Play the Coulomb pulse continuously for the whole sequence
            #      ____      ____      ____      ____
            #     |    |    |    |    |    |    |    |
            # ____|    |____|    |____|    |____|    |...
            with for_(counter, 0, counter < N, counter + 1):
                # The Coulomb pulse
                play("coulomb_step" * amp(Coulomb_amp / P1_step_amp), "P1")
                play("coulomb_step" * amp(-Coulomb_amp / P1_step_amp), "P1")

            with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
                # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
                # frequency and the integrated quadratures are stored in "I" and "Q"
                I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q)
                # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
                # is stored in "dc_signal"
                dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal)
                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
                # process them which can cause the OPX to crash.
                wait(1_000 * u.ns, "tank_circuit")
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
    simulation_config = SimulationConfig(duration=50_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, charge_stability_prog, simulation_config)
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
    job = qm.execute(charge_stability_prog)
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
        S = u.demod2volts(I + 1j * Q, reflectometry_readout_length, single_demod=True)
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
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"DC_signal_data": DC_signal})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

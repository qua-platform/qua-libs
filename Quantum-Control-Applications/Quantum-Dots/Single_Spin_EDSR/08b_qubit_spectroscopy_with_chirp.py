"""
        QUBIT SPECTROSCOPY w/ Chirp
The goal of the script is to find the qubit transition by sweeping both the qubit pulse frequency and the magnetic field.
The QUA program is divided into three sections:
    1) step between the initialization point, idle point, and the measurement point using sticky elements (long timescale).
    2) send the chirp pulse to drive the EDSR transition (short timescale).
    3) measure the state of the qubit using either RF reflectometry or dc current sensing via PSB or Elzerman readout.
A compensation pulse can be added to the long timescale sequence in order to ensure 0 DC voltage on the fast line of
the bias-tee. Alternatively one can obtain the same result by changing the offset of the slow line of the bias-tee.

In the current implementation, the magnetic field and LO frequency are being swept using the API of the relevant
instruments in Python. For this reason the OPX program is paused at each iteration, the external parameters (B and f_LO)
are updated in Python and then the QUA program is resumed to sweep the qubit intermediate frequency and measure the
state of the dot.
Also note that the qubit pulse is played at the end of the "idle" level whose duration is fixed.

Note that providing a single magnetic field value will result in acquiring the 1D qubit spectroscopy at the specified
B-field.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the DC offsets of the external DC voltage source.
    - Connecting the OPX to the fast line of the plunger gates.
    - Having calibrated the initialization and readout point from the charge stability map and updated the configuration.

Before proceeding to the next node:
    - Identify the qubit frequency and update the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool, wait_until_job_is_paused
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100

# Chirp parameters - Defined in configuration

# The intermediate frequency sweep parameters
f_min = 10 * u.MHz
f_max = 251 * u.MHz
df = 2000 * u.kHz
IFs = np.arange(f_min, f_max + 0.1, df)
# The LO frequency sweep parameters
f_min_external = 4.501e9 - f_min
f_max_external = 6.5e9 - f_max
df_external = f_max - f_min
lo_frequencies = np.arange(f_min_external, f_max_external + 0.1, df_external)
# lo_frequencies = [6e9]
# Total frequency vector
frequencies = np.array(np.concatenate([IFs + lo_frequencies[i] for i in range(len(lo_frequencies))]))

# Magnetic field in T
# B_fields = np.arange(-5, 5, 0.1)
B_fields = [0, 1, 2]

# Delay in ns before stepping to the readout point after playing the qubit pulse - must be a multiple of 4ns and >= 16ns
delay_before_readout = 16

seq = OPX_virtual_gate_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("idle", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": IFs,
    "LO_frequencies": lo_frequencies,
    "frequencies": frequencies,
    "B_fields": B_fields,
    "config": config,
}

###################
# The QUA program #
###################
with program() as qubit_spectroscopy_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    f = declare(int)  # QUA variable for the qubit pulse duration
    i = declare(int)  # QUA variable for the magnetic field sweep
    j = declare(int)  # QUA variable for the lo frequency sweep
    chirp_var = declare(int, value=chirp_rate)
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    with for_(i, 0, i < len(B_fields) + 1, i + 1):
        with for_(j, 0, j < len(lo_frequencies), j + 1):
            # pause() # Needs to be uncommented when not simulating
            with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
                with for_(*from_array(f, IFs)):  # Loop over the qubit pulse amplitude
                    update_frequency("qubit", f)

                    # Navigate through the charge stability map
                    seq.add_step(voltage_point_name="initialization")
                    seq.add_step(
                        voltage_point_name="idle", duration=chirp_duration + processing_time + delay_before_readout
                    )  # Processing time is time it takes to calculate the chirp pulse
                    seq.add_step(voltage_point_name="readout", duration=duration_readout)
                    seq.add_compensation_pulse(duration=duration_compensation_pulse)

                    # Drive the qubit by playing the MW pulse at the end of the manipulation step
                    wait((duration_init) * u.ns, "qubit")  #
                    play("chirp", "qubit", chirp=(chirp_var, chirp_units))

                    # Measure the dot right after the qubit manipulation
                    wait(
                        (duration_init + chirp_duration + processing_time + delay_before_readout) * u.ns,
                        "tank_circuit",
                        "TIA",
                    )  #
                    I, Q, I_st, Q_st = RF_reflectometry_macro()
                    dc_signal, dc_signal_st = DC_current_sensing_macro()

                    seq.ramp_to_zero()
        save(i, n_st)
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(len(IFs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(lo_frequencies)).save_all("I")
        Q_st.buffer(len(IFs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(lo_frequencies)).save_all("Q")
        # DC current sensing
        dc_signal_st.buffer(len(IFs)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(lo_frequencies)).save_all(
            "dc_signal"
        )

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, qubit_spectroscopy_prog, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    plt.figure()
    plt.subplot(211)
    samples.con1.plot()
    plt.axhline(level_init[0], color="k", linestyle="--")
    plt.axhline(level_manip[0], color="k", linestyle="--")
    plt.axhline(level_readout[0], color="k", linestyle="--")
    plt.axhline(level_init[1], color="k", linestyle="--")
    plt.axhline(level_manip[1], color="k", linestyle="--")
    plt.axhline(level_readout[1], color="k", linestyle="--")
    plt.yticks(
        [
            level_readout[1],
            level_manip[1],
            level_init[1],
            0.0,
            level_init[0],
            level_manip[0],
            level_readout[0],
        ],
        ["readout", "manip", "init", "0", "init", "manip", "readout"],
    )
    plt.legend("")
    from macros import get_filtered_voltage

    plt.subplot(212)
    get_filtered_voltage(
        job.get_simulated_samples().con1.analog["1"],
        1e-9,
        bias_tee_cut_off_frequency,
        True,
    )
    plt.show()
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
    job = qm.execute(qubit_spectroscopy_prog)
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    for i in range(len(B_fields)):  # Loop over y-voltages
        # TODO Update the magnetic field
        for j in range(len(lo_frequencies)):
            # TODO update the lo frequency
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
        progress_counter(iteration, len(B_fields))
        # Plot data
        if len(B_fields) > 1:
            plt.subplot(121)
            plt.cla()
            plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
            plt.pcolor(
                frequencies / u.MHz,
                B_fields[: iteration + 1],
                np.reshape(R, (iteration + 1, len(frequencies))),
            )
            plt.xlabel("Qubit pulse frequency [MHz]")
            plt.ylabel("B [mT]")
            plt.subplot(122)
            plt.cla()
            plt.title("Phase [rad]")
            plt.pcolor(
                frequencies / u.MHz,
                B_fields[: iteration + 1],
                np.reshape(phase, (iteration + 1, len(frequencies))),
            )
            plt.xlabel("Qubit pulse frequency [MHz]")
            plt.ylabel("B [mT]")
            plt.tight_layout()
            plt.pause(0.1)
        else:
            plt.suptitle(f"B = {B_fields[0]} mT")
            plt.subplot(121)
            plt.cla()
            plt.plot(frequencies / u.MHz, np.reshape(R, len(frequencies)))
            plt.xlabel("Qubit pulse frequency [MHz]")
            plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
            plt.subplot(122)
            plt.cla()
            plt.plot(frequencies / u.MHz, np.reshape(phase, len(frequencies)))
            plt.xlabel("Qubit pulse frequency [MHz]")
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

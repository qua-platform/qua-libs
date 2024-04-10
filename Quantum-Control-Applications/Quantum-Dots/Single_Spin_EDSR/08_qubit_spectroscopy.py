"""
        QUBIT SPECTROSCOPY
The goal of the script is to find the qubit transition by sweeping both the qubit pulse frequency and the magnetic field.
The QUA program is divided into three sections:
    1) step between the initialization point and the measurement point using sticky elements (long timescale).
    2) send the MW pulse to drive the EDSR transition (short timescale).
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


###################
# The QUA program #
###################

n_avg = 100
# The intermediate frequency sweep parameters
f_min = 1 * u.MHz
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

with program() as qubit_spectroscopy_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    f = declare(int)  # QUA variable for the qubit pulse duration
    i = declare(int)  # QUA variable for the magnetic field sweep
    j = declare(int)  # QUA variable for the lo frequency sweep
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    with for_(i, 0, i < len(B_fields) + 1, i + 1):
        with for_(j, 0, j < len(lo_frequencies), j + 1):
            pause()
            with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
                with for_(*from_array(f, IFs)):  # Loop over the qubit pulse amplitude
                    update_frequency("qubit", f)
                    with strict_timing_():  # Ensure that the sequence will be played without gap
                        # Navigate through the charge stability map
                        seq.add_step(voltage_point_name="initialization")
                        seq.add_step(voltage_point_name="readout")
                        seq.add_compensation_pulse(duration=duration_compensation_pulse)

                        # Drive the qubit by playing the MW pulse at the end of the manipulation step
                        wait((duration_init - delay_before_readout - cw_len) * u.ns, "qubit")
                        play("cw", "qubit")

                        # Measure the dot right after the qubit manipulation
                        wait(duration_init * u.ns, "tank_circuit", "TIA")
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
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, qubit_spectroscopy_prog, simulation_config)
    # Plot the simulated samples
    plt.figure()
    plt.subplot(211)
    job.get_simulated_samples().con1.plot()
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
    get_filtered_voltage(job.get_simulated_samples().con1.analog["1"], 1e-9, bias_tee_cut_off_frequency, True)

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
        S = u.demod2volts(I + 1j * Q, reflectometry_readout_length)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        DC_signal = u.demod2volts(DC_signal, readout_len)
        # Progress bar
        progress_counter(iteration, len(B_fields))
        # Plot data
        if len(B_fields) > 1:
            plt.subplot(121)
            plt.cla()
            plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
            plt.pcolor(frequencies / u.MHz, B_fields[: iteration + 1], np.reshape(R, (iteration + 1, len(frequencies))))
            plt.xlabel("Qubit pulse frequency [MHz]")
            plt.ylabel("B [mT]")
            plt.subplot(122)
            plt.cla()
            plt.title("Phase [rad]")
            plt.pcolor(
                frequencies / u.MHz, B_fields[: iteration + 1], np.reshape(phase, (iteration + 1, len(frequencies)))
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
